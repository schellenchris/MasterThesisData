import json
import time
from pickletools import dis

import SimpleITK as sitk
import voxelmorph as vxm
from tensorflow.keras.callbacks import EarlyStopping
from voxelmorph.tf.utils import point_spatial_transformer
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler

from ImageRegistrationInterface import ImageRegistrationInterface
import vxmlosses
import math
from tensorflow import keras

# Seed value
# Apparently you may use different seed values at each stage
seed_value = 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_GPU_ALLOCATOR']= 'cuda_malloc_async'

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.random.set_seed(seed_value)

initial_learning_rate = 0

class SaveModelCallback(keras.callbacks.Callback):
    def __init__(self, output_path) -> None:
        super().__init__()
        self.output_path = output_path
        self.best_loss = 50

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(os.path.join(self.output_path, 'weightsCheckpoint.h5'))
        if self.best_loss > logs.get("val_loss"):
            self.best_loss = logs.get("val_loss")
            print(f"\nBest model saved, val_loss: {self.best_loss}")
            self.model.save_weights(os.path.join(self.output_path, 'bestWeights.h5'))


def lr_exp_decay(epoch, lr):
    k = 0.1
    epoch /= 10
    return initial_learning_rate #* math.exp(-k*epoch)

def train_vxm_model(train_generator,
                    val_generator,
                    multi_gpu=False,
                    epochs: int = 300,
                    steps_per_epoch: int = 100,
                    learning_rate: float = 0.001,
                    loss: str = 'MSE',
                    model_name: str = 'default_name',
                    dataset: str = 'mnist',
                    nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]],
                    int_resolution = 2
                    ) -> str:
    global initial_learning_rate 
    initial_learning_rate = learning_rate 
    start_time = time.time()
    # Define Tensorboard as a Keras callback
    log_dir = f'../runs/logs/{dataset}/vmxtf_{model_name}'
    tensorboard = TensorBoard(log_dir=log_dir,
                              histogram_freq=1,
                              write_images=True)
    in_sample, out_sample = next(train_generator)
    fixed_images = in_sample[1]
    fixed_image = fixed_images[0]
    if multi_gpu:
        device_type = 'GPU'
        devices = tf.config.experimental.list_physical_devices(device_type)
        devices_names = [d.name.split('e:')[1] for d in devices]
        #single gpu
        devices_names = [devices_names[0]]
        
        strategy = tf.distribute.MirroredStrategy(devices=devices_names)
        with strategy.scope():
            vxm_model = get_vxm_model(sitk.GetImageFromArray(fixed_image),
                                      loss=loss,
                                      learning_rate=learning_rate,
                                      nb_features=nb_features,
                                      int_resolution=int_resolution)
    else:
        vxm_model = get_vxm_model(sitk.GetImageFromArray(fixed_image),
                                  loss=loss,
                                  learning_rate=learning_rate,
                                  nb_features=nb_features)
    output_path = os.path.join('/home/cschellenberger/Documents/scripts/models', dataset, model_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    hist = vxm_model.fit(train_generator,
                         validation_data=val_generator,
                         validation_steps = 4,
                         epochs=epochs,
                         steps_per_epoch=steps_per_epoch,
                         verbose=1,
                         callbacks=[tensorboard, LearningRateScheduler(lr_exp_decay, verbose=1), SaveModelCallback(output_path)]) #early_stopping
    end_time = time.time()
    #model_name = f'vxmtf_{model_name}{f"_actual_ep{len(hist.epoch)}" if len(hist.epoch) != epochs else ""}_final_loss{str(final_loss).replace(".", "_")}'
    print(f'Finished training model was save to {output_path}')
    json.dump(end_time - start_time,
              open(os.path.join(output_path, 'train_time.json'), 'w'))
    #json.dump(hist.history, open(os.path.join(output_path, 'hist.json'), 'w'))
    vxm_model.save_weights(os.path.join(output_path, 'weights.h5'))
    return os.path.join(output_path, 'weights.h5')


def get_vxm_model(fixed_image, loss='MSE', learning_rate=0.001, nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]], int_resolution = 2):
    fixed_image_np = sitk.GetArrayFromImage(fixed_image)
    # configure unet input shape (concatenation of moving and fixed images)
    inshape = fixed_image_np.shape
    # build model
    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=7, int_resolution=int_resolution) #, int_resolution=2
    # voxelmorph has a variety of custom loss classes
    if loss == 'MSE':
        losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    elif loss == 'NCC':
        losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2', loss_mult=int_resolution).loss]
    elif loss == 'NMIDEV':
        losses = [vxm.losses.MutualInformation(nb_bins=32).loss]
    elif loss == 'NMI':
        losses = [vxmlosses.NMI(bin_centers = np.linspace(0, 1, num=32), vol_size = (128, 256, 256), patch_size = 16, max_clip=np.inf, local = True, crop_background=True).loss, vxm.losses.Grad('l2', loss_mult = int_resolution).loss] #, vxm.losses.Grad('l2', loss_mult = int_resolution).loss
    else:
        raise NotImplementedError(
            f'{loss} is not implemented yet please select one of the following losses [MSE, NCC]'
        )
    # usually, we have to balance the two losses by a hyper-parameter
    loss_weights = [1, 0.08] #[1, 0.4]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #weights_path = '/home/cschellenberger/Documents/scripts/models/synthetic/newT3_best_localmi_reg001_08_1000_st43_lr3e-05_bat1/bestWeights.h5'
    #vxm_model.load_weights(weights_path)
    vxm_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)#loss_weights=loss_weights
    return vxm_model


class VoxelmorphTF(ImageRegistrationInterface):
    def __init__(self, weights_path, fixed_image: sitk.Image, nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]], int_resolution = 2):
        super().__init__()
        fixed_image_np = sitk.GetArrayFromImage(fixed_image)
        self.model = get_vxm_model(fixed_image,
            loss='MSE',
            learning_rate=0.01,
            nb_features=nb_features,
            int_resolution=int_resolution)
        self.model.load_weights(weights_path)

    def register_images(self, moving_image: sitk.Image, fixed_image: sitk.Image, device = '/GPU'):
        moving_image_np = sitk.GetArrayFromImage(moving_image)[np.newaxis, :]
        fixed_image_np = sitk.GetArrayFromImage(fixed_image)[np.newaxis, :]
        start_time = time.time()
        with tf.device(device):
            moved_image_np, displacement_np = self.model.predict([moving_image_np, fixed_image_np])
        end_time = time.time()
        moved_image = sitk.GetImageFromArray(moved_image_np.squeeze())
        moved_image.SetSpacing(fixed_image.GetSpacing())
        moved_image.SetOrigin(fixed_image.GetOrigin())
        return moved_image, displacement_np, end_time - start_time

    @staticmethod
    def get_moved_points(fixed_landmarks: np.array,
                         displacement: sitk.Image) -> np.array:
        fixed_landmarks = fixed_landmarks[np.newaxis, ...].astype('float32')
        displacement_np = sitk.GetArrayFromImage(displacement)
        displacement_np = np.swapaxes(displacement_np, 0, 2)
        displacement_np = displacement_np[np.newaxis, ...].astype('float32')
        # obtain moved_image_landmarks
        moved_landmarks = point_spatial_transformer(
            (tf.convert_to_tensor(fixed_landmarks),
             tf.convert_to_tensor(displacement_np)))
        # moved_landmarks convert into ndarray
        moved_landmarks = np.squeeze(moved_landmarks)
        return moved_landmarks