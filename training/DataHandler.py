import os
import random

import SimpleITK as sitk
import numpy as np
from tensorflow.keras.datasets import mnist


class DataHandler:
    def __init__(self, val_images: int = 12):
        self.x_train = self.y_train = np.ndarray((0, 0))
        self.x_val = self.y_val = np.ndarray((0, 0))
        self.val_images = val_images

    def get_mnist_data(self, select_number=None) -> None:
        (x_train_load, y_train_load), _ = mnist.load_data()
        # Only use the images from the selected number
        if select_number:
            self.x_train = x_train_load[y_train_load == select_number, ...]
            self.y_train = y_train_load[y_train_load == select_number]
        else:
            self.x_train = x_train_load
            self.y_train = y_train_load

        # double the val_images because we have no image pairs
        val_images = self.val_images * 2
        # split data into train and validation
        self.x_val = self.x_train[-val_images:, ...]
        self.y_val = self.y_train[-val_images:]
        self.x_train = self.x_train[:-val_images, ...]
        self.y_train = self.y_train[:-val_images]
        # normalize values to [0, 1]
        self.x_train = self.x_train.astype('float') / 255
        self.y_train = self.y_train.astype('float') / 255
        self.x_val = self.x_val.astype('float') / 255
        self.y_val = self.y_val.astype('float') / 255

        # pad the images so they have the size 32x32
        pad_amount = ((0, 0), (2, 2), (2, 2))
        self.x_train = np.pad(self.x_train, pad_amount, 'constant')
        self.x_val = np.pad(self.x_val, pad_amount, 'constant')

    @staticmethod
    def __get_all_file_paths_in_path__(path: str,
                                       traverse_sub_dir: bool = False
                                       ) -> np.array:
        # all files listed in this array will be excluded can be used for corrupted files etc.
        exclude = 'Model117'
        abs_file_paths = np.ndarray((0, 0))
        for dir_path, dirs, filenames in os.walk(path):
            if not traverse_sub_dir:
                # this is the only way to skip directories in os.walk
                dirs = []

            filenames.sort()
            for f in filenames:
                path = os.path.join(dir_path, f)
                if exclude not in path:
                    abs_file_paths = np.append(abs_file_paths, path)
        return abs_file_paths

    @staticmethod
    def __random_resample_image__(image: sitk.Image) -> np.array:
        transforms = []
        if random.random() < 0.5:
            dimension = image.GetDimension()
            x_shear = np.random.uniform(-0.2, 0.2)
            y_shear = np.random.uniform(-0.2, 0.2)
            transform = sitk.AffineTransform(dimension)
            matrix = np.array(transform.GetMatrix()).reshape(
                (dimension, dimension))
            matrix[0, 1] = -x_shear
            matrix[1, 0] = -y_shear
            transform.SetMatrix(matrix.ravel())
            transforms.append(transform)

        if random.random() < 0.5:
            rand_rotation = random.uniform(-0.1745, 0.1745)
            transform = sitk.VersorTransform((0, 0, 1), rand_rotation)
            center_index = [x / 2 for x in image.GetSize()]
            transform.SetCenter(
                image.TransformContinuousIndexToPhysicalPoint(center_index))
            transforms.append(transform)

        if len(transforms) > 0:
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(image)
            composite_transform = sitk.Transform(3, sitk.sitkEuler)
            for transform in transforms:
                composite_transform.AddTransform(transform)
            resampler.SetTransform(composite_transform)
            return sitk.GetArrayFromImage(resampler.Execute(image))
        return sitk.GetArrayFromImage(image)

    def get_synthetic_data(self,
                           fixed_path: str,
                           moving_path: str,
                           traverse_sub_dir: bool = False) -> None:
        moving_paths = self.__get_all_file_paths_in_path__(
            moving_path, traverse_sub_dir=traverse_sub_dir)
        [self.x_val, self.x_train] = np.split(moving_paths, [self.val_images],
                                              axis=0)

        fixed_paths = self.__get_all_file_paths_in_path__(
            fixed_path, traverse_sub_dir=traverse_sub_dir)
        [self.y_val, self.y_train] = np.split(fixed_paths, [self.val_images],
                                              axis=0)

    def add_synthetic_data(self,
                           fixed_path: str,
                           moving_path: str,
                           traverse_sub_dir: bool = False) -> None:
        self.x_train = np.append(self.x_train, self.__get_all_file_paths_in_path__(
            moving_path, traverse_sub_dir=traverse_sub_dir))
        self.y_train = np.append(self.y_train, self.__get_all_file_paths_in_path__(
            fixed_path, traverse_sub_dir=traverse_sub_dir))

    def data_gen_mnist(self, data, batch_size=4):
        data_shape = data.shape[1:]
        num_dimensions = len(data_shape)
        zero_phi = np.zeros([batch_size, *data_shape, num_dimensions])

        while True:
            idx1 = np.random.randint(0, data.shape[0], size=batch_size)
            moving_images = data[idx1, ..., np.newaxis]
            idx2 = np.random.randint(0, data.shape[0], size=batch_size)
            fixed_images = data[idx2, ..., np.newaxis]

            inputs = [moving_images, fixed_images]
            outputs = [fixed_images, zero_phi]

            yield inputs, outputs

    def data_gen_sitk_images(self, paths) -> sitk.Image:
        idx = 0
        while True:
            if idx > len(paths):
                idx = 0
            image = sitk.ReadImage(paths[idx])
            yield image
            idx += 1

    def data_gen_voxelmorph(
        self,
        data_x,
        data_y,
        random_resampling: bool = False,
        batch_size=4,
        shuffle=False,
    ):
        idx = 0
        allIdx = np.array([i for i in range(len(data_x))])
        while True:
            if idx + batch_size > len(data_x):
                idx = 0
            if shuffle:
                if len(allIdx) < batch_size:
                    allIdx = np.array([i for i in range(len(data_x))])
                idxs = np.random.randint(0, len(allIdx), size=batch_size)
                idxsTmp = allIdx.take(idxs)
                allIdx = np.delete(allIdx, idxs)
                idxs = idxsTmp
                idxs_fixed = np.random.randint(0,
                                               data_x.shape[0],
                                               size=batch_size)
            else:
                idxs = range(idx, idx + batch_size)
                idxs_fixed = [idx + self.val_images for idx in idxs]
            if type(data_x[0]) is np.ndarray:
                moving_images = data_x[idxs, ...]
                fixed_images = data_y[idxs_fixed, ...]
            elif type(data_x[0]) is np.str_:
                moving_paths = data_x[idxs, ...]
                moving_images = np.array([
                    sitk.GetArrayFromImage(sitk.ReadImage(path))
                    for path in moving_paths
                ])
                fixed_paths = data_y[idxs, ...]
                fixed_images = np.array([
                    sitk.GetArrayFromImage(sitk.ReadImage(path))
                    for path in fixed_paths
                ])
            else:
                raise TypeError(
                    'data_x is not a valid type it has to be array[str] or array[np.array]'
                )

            data_shape = fixed_images[0].shape
            num_dimensions = len(data_shape)
            zero_phi = np.zeros([batch_size, *data_shape, num_dimensions])
            if random_resampling:
                for idx in range(len(moving_images)):
                    moving_image = sitk.GetImageFromArray(moving_images[idx,
                                                                        ...])
                    moving_images[idx, ...] = self.__random_resample_image__(
                        moving_image)
                if random.random() < 0.5:
                    inputs = [fixed_images, moving_images]
                    outputs = [moving_images, zero_phi]
                else:
                    inputs = [moving_images, fixed_images]
                    outputs = [fixed_images, zero_phi]
            else:
                inputs = [moving_images, fixed_images]
                outputs = [fixed_images, zero_phi]
            yield inputs, outputs

    def data_gen_val(self, data_x, data_y):
        idx = 0
        while True:
            if idx > len(data_x):
                idx = 0
            if type(data_x[0]) is np.ndarray:
                moving_image = sitk.GetImageFromArray(data_x[idx])
                fixed_image = sitk.GetImageFromArray(data_x[idx +
                                                            self.val_images])
            elif type(data_x[0]) is str:
                moving_image = sitk.ReadImage(data_x[idx])
                fixed_image = sitk.ReadImage(data_y[idx])
            else:
                raise TypeError(
                    'data_x is not a valid type it has to be array[str] or array[np.array]'
                )
            yield moving_image, fixed_image
            idx += 1