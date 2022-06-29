import json
import time
from pickletools import dis

import SimpleITK as sitk
import voxelmorph as vxm
from tensorflow.keras.callbacks import EarlyStopping
from voxelmorph.tf.utils import point_spatial_transformer
from tensorflow.keras.callbacks import TensorBoard

from ImageRegistrationInterface import ImageRegistrationInterface

# Seed value
# Apparently you may use different seed values at each stage
seed_value = 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.random.set_seed(seed_value)


def train_vxm_model(train_generator,
                    val_generator,
                    multi_gpu=False,
                    epochs: int = 300,
                    steps_per_epoch: int = 100,
                    learning_rate: float = 0.001,
                    loss: str = 'MSE',
                    model_name: str = 'default_name',
                    dataset: str = 'mnist',
                    nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]
                    ) -> str:
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
        strategy = tf.distribute.MirroredStrategy(devices=devices_names)
        with strategy.scope():
            vxm_model = get_vxm_model(sitk.GetImageFromArray(fixed_image),
                                      loss=loss,
                                      learning_rate=learning_rate,
                                      nb_features=nb_features)
    else:
        vxm_model = get_vxm_model(sitk.GetImageFromArray(fixed_image),
                                  loss=loss,
                                  learning_rate=learning_rate,
                                  nb_features=nb_features)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   restore_best_weights=True,
                                   patience=50)
    val_data = next(val_generator)
    vxm_model.summary()
    hist = vxm_model.fit(train_generator,
                         validation_data=val_data,
                         validation_batch_size=4,
                         epochs=epochs,
                         steps_per_epoch=steps_per_epoch,
                         verbose=1,
                         callbacks=[early_stopping, tensorboard])
    final_loss = "{:.4f}".format(hist.history['loss'][-1])
    end_time = time.time()
    model_name = f'vxmtf_{model_name}{f"_actual_ep{len(hist.epoch)}" if len(hist.epoch) != epochs else ""}_final_loss{str(final_loss).replace(".", "_")}'
    output_path = os.path.join(
        '/home/cschellenberger/Documents/scripts/models', dataset,
        model_name)
    print(f'Finished training model was save to {output_path}')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    json.dump(end_time - start_time,
              open(os.path.join(output_path, 'train_time.json'), 'w'))
    json.dump(hist.history, open(os.path.join(output_path, 'hist.json'), 'w'))
    vxm_model.save_weights(os.path.join(output_path, 'weights.h5'))
    return os.path.join(output_path, 'weights.h5')


def get_vxm_model(fixed_image, loss='MSE', learning_rate=0.001, nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]):
    fixed_image_np = sitk.GetArrayFromImage(fixed_image)
    # configure unet input shape (concatenation of moving and fixed images)
    inshape = fixed_image_np.shape
    # configure unet features

    # nb_features = [
    #     [16, 16, 32, 32],    # encoder features
    #     [32, 32, 16, 16]     # decoder features
    #     # [16, 32, 32, 32],    # encoder features
    #     # [32, 32, 32, 32, 32, 16, 16] 
    # ]

    # build model
    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
    # voxelmorph has a variety of custom loss classes
    if loss == 'MSE':
        losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    elif loss == 'NCC':
        losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]
    elif loss == 'NMI':
        losses = [vxm.losses.MutualInformation().loss]
    else:
        raise NotImplementedError(
            f'{loss} is not implemented yet please select one of the following losses [MSE, NCC]'
        )

    # usually, we have to balance the two losses by a hyper-parameter
    loss_weights = [1, 0.05]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    vxm_model.compile(optimizer=optimizer,
                      loss=losses,
                      loss_weights=loss_weights)
    return vxm_model


class VoxelmorphTF(ImageRegistrationInterface):
    @staticmethod
    def register_images(
            fixed_image: sitk.Image,
            moving_image: sitk.Image,
            weights_path='/home/cschellenberger/Documents/scripts/models/vxm_mnist.index',
            nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]
            ) -> {sitk.Image, sitk.Image, int}:
        moving_image_np = sitk.GetArrayFromImage(moving_image)[np.newaxis, :]
        fixed_image_np = sitk.GetArrayFromImage(fixed_image)[np.newaxis, :]
        with tf.device('/cpu:0'):
            model = get_vxm_model(fixed_image, nb_features=nb_features)
            model.load_weights(weights_path)
            start_time = time.time()
            moved_image_np, displacement_np = model.predict(
                [moving_image_np, fixed_image_np])
            end_time = time.time()
        moved_image = sitk.GetImageFromArray(moved_image_np.squeeze(0))
        moved_image.SetSpacing(fixed_image.GetSpacing())
        moved_image.SetOrigin(fixed_image.GetOrigin())
        moved_image.SetDirection(fixed_image.GetDirection())
        displacement = displacement_np.squeeze(0)
        #displacement = np.swapaxes(displacement, 1, 2)
        displacement = sitk.GetImageFromArray(displacement, isVector=True)
        displacement.SetSpacing(fixed_image.GetSpacing())
        displacement.SetOrigin(fixed_image.GetOrigin())
        displacement.SetDirection(fixed_image.GetDirection())
        return moved_image_np, displacement_np, end_time - start_time

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