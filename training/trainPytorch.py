import json
import os

from scipy.constants import value
from torch import nn
from ImageRegistrationInterface import ImageRegistrationInterface

os.environ['VXM_BACKEND'] = 'pytorch'
import time
import numpy as np
import torch as th
import voxelmorph as vxm
import SimpleITK as sitk
from torch.utils.tensorboard import SummaryWriter
import pytorchlosses


class VoxelmorphTorch(ImageRegistrationInterface):
    def __init__(self, weights_path, fixed_image: sitk.Image, nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]):
        super().__init__()
        self.device = th.device('cuda')
        fixed_image = sitk.GetArrayFromImage(fixed_image)
        inshape = fixed_image.shape
        self.model = nn.DataParallel(
            vxm.networks.VxmDense(inshape=inshape,
                                nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]],
                                bidir=False,
                                int_steps=0,
                                int_downsize=2))
        state_dict = th.load(weights_path)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def register_images(self, moving_image, fixed_image, gpu: str = ''):
        # configure unet input shape (concatenation of moving and fixed images)
        fixed_image = sitk.GetArrayFromImage(fixed_image)
        moving_image = sitk.GetArrayFromImage(moving_image)
        fixed_image = fixed_image[np.newaxis, :]
        moving_image = moving_image[np.newaxis, :]
        device = th.device(f'cuda')
        fixed_image_th = th.from_numpy(fixed_image).to(device).float().unsqueeze(1)
        moving_image_th = th.from_numpy(moving_image).to(device).float().unsqueeze(1)
        y_pred = self.model(moving_image_th, fixed_image_th)
        return y_pred[0], y_pred[1], 0

class FullModel(nn.Module):
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, *inputs):
    outputs = self.model(*inputs)
    loss = self.loss(inputs[1], outputs[0])
    return th.unsqueeze(loss,0), outputs
    
def train_vxm_model(train_generator,
                    val_generator,
                    epochs: int = 10,
                    steps_per_epoch: int = 100,
                    gpu: str = '',
                    multi_gpu: bool = False,
                    learning_rate: float = 0.001,
                    loss: str = 'MSE',
                    model_name: str = 'default_name',
                    dataset: str = 'mnist',
                    nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]) -> str:
    start_time = time.time()
    model_dir = f'../models/{dataset}/'
    in_sample, out_sample = next(train_generator)
    fixed_images = in_sample[1]
    fixed_image = fixed_images[0]
    # configure unet input shape (concatenation of moving and fixed images)
    inshape = fixed_image.shape

    device = th.device(f'cuda{gpu}')
    int_downsize = 2
    model = nn.DataParallel(FullModel(vxm.networks.VxmDense(inshape=inshape,
                              nb_unet_features=nb_features,
                              int_steps=0), pytorchlosses.MutualInformation(num_bin=32))).cuda()

    # prepare the model for training and send to device
    #model.to(device)

    # set optimizer
    optimizer = th.optim.Adam(model.parameters(),
                              lr=learning_rate,
                              amsgrad=False,
                              eps=1e-7,
                              betas=(0.9, 0.999),
                              weight_decay=0)
    if len(fixed_image.shape) == 2:
        grad_loss = vxm.losses.Grad2D('l2', loss_mult=int_downsize).loss
    else:
        assert len(
            fixed_image.shape
        ) == 3, f'grad is only implemented for 2D and 3D but got {len(fixed_image.shape)}'
        grad_loss = vxm.losses.Grad('l2', loss_mult=int_downsize).loss
    if loss == 'MSE': losses = [vxm.losses.MSE().loss, grad_loss]
    elif loss == 'NMI': losses = [pytorchlosses.MutualInformation(num_bin=32).forward]
    else: raise NotImplementedError(
            f'{loss} is not implemented yet please select one of the following losses [MSE, NMI]'
        )
    weights = [1]

    best_model = model
    best_loss = 100
    train_writer = SummaryWriter(f'../runs/logs/{dataset}/vxmth_{model_name}/train')
    val_writer = SummaryWriter(f'../runs/logs/{dataset}/vxmth_{model_name}/validation')
    # training loops, TODO in step function
    for epoch in range(0, epochs):
        #best_model.save(os.path.join(model_dir, 'best_model.pt'))
        
        model.train(True)
        epoch_loss = []
        epoch_total_loss = []

        for step in range(steps_per_epoch):
            step_start_time = time.time()
            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(train_generator)
            inputs = [
                th.from_numpy(image).to(device).float().unsqueeze(1)
                for image in inputs
            ]
            y_true = [
                th.from_numpy(image).to(device).float().unsqueeze(1)
                for image in y_true
            ]
            #y_true[1] = y_true[1].permute(0, 5, 2, 3, 4, 1)
            #y_true[1] = y_true[1].squeeze(5)
            # run inputs through the model to produce a warped image and flow field
            loss, _ = model(*inputs)
            loss = loss.sum()
            loss_list = []
            #loss = losses[0](y_true[0], y_pred[0])
            loss_list.append('%.6f' % loss.item())

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())
            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print step info
            epoch_info = 'epoch: %04d' % (epoch + 1)
            step_info = ('step: %d/%d' % (step + 1, steps_per_epoch)).ljust(14)
            time_info = 'time: %.2f sec' % (time.time() - step_start_time)
            epoch_losses = [
                '%.6f' % f
                for f in np.mean(np.array(epoch_loss, dtype=np.float), axis=0)
            ]
            losses_info = ', '.join(epoch_losses)
            loss_info = 'loss: %.6f  (%s)' % (np.mean(epoch_total_loss), losses_info)
            print(' '.join((epoch_info, step_info, time_info, loss_info)), flush=True)

        model.train(False)
        train_writer.add_scalar('epoch_loss', np.mean(epoch_total_loss), epoch)
        train_writer.add_scalar('epoch_flow_loss', float(epoch_losses[0]), epoch)

        # calculate the total val loss
        del loss, inputs, y_true
        loss_list = []
        epoch_loss = []
        epoch_total_loss = []
        val_loss = 0
        val_loss_list = []
        val_epoch_loss = []
        val_epoch_total_loss = []
        with th.no_grad():
            for idx in range(2):
                inputs, y_true = next(val_generator)
                inputs = [
                    th.from_numpy(image).to(device).float().unsqueeze(1)
                    for image in inputs
                ]
                y_true = [
                    th.from_numpy(image).to(device).float().unsqueeze(1)
                    for image in y_true
                ]
                # run inputs through the model to produce a warped image and flow field
                val_loss, _ = model(*inputs)
                val_loss = val_loss.sum()
                #val_loss = losses[0](y_true[0], y_pred[0])
                val_loss_list.append('%.6f' % val_loss.item())
                val_epoch_loss.append(val_loss_list)
                val_epoch_total_loss.append(val_loss.item())
                loss_info = 'val loss: %.6f' % (np.mean(val_epoch_total_loss))
                print(' '.join((epoch_info, step_info, time_info, loss_info)), flush=True)


        val_epoch_losses = [
            '%.6f' % f
            for f in np.mean(np.array(val_epoch_loss, dtype=np.float), axis=0)
        ]
        val_writer.add_scalar('epoch_loss', np.mean(val_epoch_total_loss),
                              epoch)
        val_writer.add_scalar('epoch_flow_loss', float(val_epoch_losses[0]),
                              epoch)
        # val_writer.add_scalar('epoch_transformer_loss',
        #                       float(val_epoch_losses[1]), epoch)
        # train_writer.add_scalar('epoch_transformer_loss',
        #                         float(epoch_losses[1]), epoch)
    end_time = time.time()
    model_name = f'vxmth_{model_name}_final_loss{str(np.mean(epoch_total_loss)).replace(".", "_")}'
    output_path = os.path.join('/home/cschellenberger/Documents/scripts/models', dataset, model_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    json.dump(end_time - start_time,
              open(os.path.join(output_path, 'train_time.json'), 'w'))
    # final model save
    th.save(model.module.state_dict(), os.path.join(output_path, 'model.pt'))
    return os.path.join(output_path, 'model.pt')