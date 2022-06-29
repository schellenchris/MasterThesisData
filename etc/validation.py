import json
import os

from DataHandler import DataHandler
#from frameworks.Airlab import Airlab
#from frameworks.SimpleElastix import SimpleElastix
import SimpleITK as sitk
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns


def meshgridnd_like(in_img, rng_func=range):
    new_shape = list(in_img.shape)
    all_range = [rng_func(i_len) for i_len in new_shape]
    return tuple([x_arr.swapaxes(0, 1) for x_arr in np.meshgrid(*all_range)])


def get_displacement_3d(displacement):
    displacement_np = sitk.GetArrayFromImage(displacement)
    DS_FACTOR = 16
    c_xx, c_yy, c_zz = [
        x.flatten() for x in meshgridnd_like(
            displacement_np[::DS_FACTOR, ::DS_FACTOR, ::DS_FACTOR, 0])
    ]

    get_flow = lambda i: displacement_np[::DS_FACTOR, ::DS_FACTOR, ::DS_FACTOR,
                                         i].flatten()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')

    ax.quiver(c_xx,
              c_yy,
              c_zz,
              get_flow(0),
              get_flow(1),
              get_flow(2),
              length=0.5,
              normalize=True)


def get_landmarks(fixed_image_path: str,
                  indexing: str = 'zyx') -> (np.array, np.array):
    model_name = os.path.basename(fixed_image_path).replace('_atn_3.nrrd', '')
    loaded_points = np.load(
        f'/home/lschilling/datam2olie/synthetic/orig/CT_points_t1_t3/{model_name}_idx.npz'
    )
    moving_landmarks = loaded_points['t1']
    fixed_landmarks = loaded_points['t3']
    if indexing == 'zyx':
        # swap columns because numpy and vxm use zyx indexing and the data uses xyz indexing
        moving_landmarks[:, [0, 2]] = moving_landmarks[:, [2, 0]]
        fixed_landmarks[:, [0, 2]] = fixed_landmarks[:, [2, 0]]
    if indexing == 'yxz':
        # swap columns because numpy and vxm use zyx indexing and the data uses xyz indexing
        moving_landmarks[:, [0, 1]] = moving_landmarks[:, [1, 0]]
        fixed_landmarks[:, [0, 1]] = fixed_landmarks[:, [1, 0]]
    else:
        assert indexing == 'xyz', f'indexing can only be xyz or zyx. Got: {indexing}'
    return moving_landmarks.astype(np.float64), fixed_landmarks.astype(
        np.float64)


def get_tre(moving_landmarks: np.array, moved_landmarks: np.array) -> np.array:
    differences = moved_landmarks - moving_landmarks
    distances_array = np.linalg.norm(differences, axis=1)
    return distances_array


def get_tre_non_reg(moving_landmarks: np.array,
                    fixed_landmarks: np.array) -> np.array:
    differences = fixed_landmarks - moving_landmarks
    distances_array = np.linalg.norm(differences, axis=1)
    return distances_array


def get_mse(moved_image: sitk.Image, fixed_image: sitk.Image) -> float:
    moved_image_np = sitk.GetArrayFromImage(moved_image).astype(np.float64)
    fixed_image_np = sitk.GetArrayFromImage(fixed_image).astype(np.float64)
    difference = np.subtract(moved_image_np, fixed_image_np)
    squared_difference = np.square(difference)
    mse = squared_difference.mean()
    return mse


def get_jacobian_np(displacement: sitk.Image) -> np.array:
    jacobian_filter = sitk.DisplacementFieldJacobianDeterminantFilter()
    jacobian_np = sitk.GetArrayFromImage(jacobian_filter.Execute(displacement))
    return jacobian_np


def get_moved_points(points: np.array, displacement: sitk.Image) -> np.array:
    displacement_transform = sitk.DisplacementFieldTransform(sitk.Cast(displacement, sitk.sitkVectorFloat64))
    moved_points = [
        displacement_transform.TransformPoint(point) for point in points
    ]
    return moved_points


def get_moved_fixed_checkerboard(framework, moving_image_path,
                                 fixed_image_path):
    moving_image = sitk.ReadImage(moving_image_path)
    fixed_image = sitk.ReadImage(fixed_image_path)
    moved_image, displacement, time = framework.register_images(
        fixed_image=fixed_image,
        moving_image=moving_image,
        weights_path=model_path,
        loss=loss)
    checkerboard = sitk.CheckerBoard(moved_image, fixed_image, [10, 10, 1])
    return checkerboard


framework_name = 'vxmtf'
dataset = 'synthetic'
model_path = '/home/cschellenberger/Documents/scripts/models/synthetic/lucavxm/weights.h5'
model_name = os.path.basename(model_path)
loss = 'MSE'
#if framework_name == 'vxmtf':
from TrainVoxelmorph import VoxelmorphTF
    #from frameworks.VoxelmorphTorch import VoxelmorphTorch
#else:
    #from frameworks.VoxelmorphTorch import VoxelmorphTorch
    #from frameworks.VoxelmorphTF import VoxelmorphTF

frameworks = {
    'vxmtf': VoxelmorphTF(),
    #'airlab': Airlab(),
    #'simpleelastix': SimpleElastix(),
    #'vxmth': VoxelmorphTorch()
}
framework = frameworks[framework_name]

dh = DataHandler(val_images=12)
dh.get_synthetic_data(
    fixed_path='/home/cschellenberger/datam2olie/synthetic/orig/t3/Synthetic_CT/',
    moving_path='/home/cschellenberger/datam2olie/synthetic/orig/t1/Synthetic_CT/',
    traverse_sub_dir=False)
moving_image_paths = dh.x_val
fixed_image_paths = dh.y_val
if dataset == 'synthetic':
    tre_list = []
    tre_non_reg_list = []
time_list = []
mse_list = []
mse_non_reg_list = []
jacobian_reflections_list = []
checkerboard = None
for (idx, _) in enumerate(moving_image_paths):
    moving_image = sitk.ReadImage(moving_image_paths[idx])
    fixed_image = sitk.ReadImage(fixed_image_paths[idx])
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moved_image, displacement, time = framework.register_images(
        fixed_image=fixed_image,
        moving_image=moving_image,
        weights_path=model_path,
    )
    displacement = displacement.squeeze()
    displacement[:,:,:,[0,2]] = displacement[:,:,:,[2,0]]
    #displacement = 1.8 * displacement
    moved_image = moved_image.squeeze()
    moved_image = sitk.GetImageFromArray(moved_image)
    displacement = sitk.GetImageFromArray(displacement, isVector=True)
    displacement.SetSpacing(fixed_image.GetSpacing())
    displacement.SetOrigin(fixed_image.GetOrigin())
    displacement.SetDirection(fixed_image.GetDirection())
    moved_image.SetSpacing(fixed_image.GetSpacing())
    moved_image.SetOrigin(fixed_image.GetOrigin())
    moved_image.SetDirection(fixed_image.GetDirection())
    jacobian = get_jacobian_np(displacement)
    jacobian_reflections_list.append(len(jacobian[jacobian < 0]))
    time_list.append(time)
    mse_list.append(get_mse(moved_image, fixed_image))
    mse_non_reg_list.append(get_mse(moving_image, fixed_image))
    # if idx == 0:
    #     checkerboard = sitk.CheckerBoard(moved_image, fixed_image, [10, 10])
    #     sitk.Show(checkerboard)
    if dataset == 'synthetic':
        moving_landmarks, fixed_landmarks = get_landmarks(
            fixed_image_paths[idx], indexing='xyz')
        moved_landmarks = get_moved_points(fixed_landmarks, displacement)
        tre = get_tre(moving_landmarks, moved_landmarks)
        tre_non_reg = get_tre_non_reg(moving_landmarks, fixed_landmarks)
        tre_list.append(tre.mean())
        tre_non_reg_list.append(tre_non_reg.mean())
        pass

output_path = os.path.dirname(model_path)
if not os.path.exists(output_path):
    os.mkdir(output_path)

#sitk.WriteImage(checkerboard,
#                os.path.join(output_path, 'result_checkerboard.nrrd'))
mse_list_np = np.array(mse_list)

results_dict = {
    'time_list': time_list,
    'time_mean': np.array(time_list).mean(),
    'mse_non_reg_list': mse_non_reg_list,
    'mse_non_reg_mean': np.array(mse_non_reg_list).mean(dtype=float),
    'jacobian_reflections_list': jacobian_reflections_list,
    'jacobian_reflections_mean': np.array(jacobian_reflections_list).mean(),
    'mse_list': mse_list,
    'mse_mean': mse_list_np.mean(dtype=float),
    'mse_min': np.min(mse_list_np),
    'mse_max': np.max(mse_list_np),
    'mse_std': np.std(mse_list_np),
    'mse_25': np.percentile(mse_list_np, 25),
    'mse_50': np.percentile(mse_list_np, 50),
    'mse_75': np.percentile(mse_list_np, 75),
    'mse_var': np.var(mse_list_np)
}

if dataset == 'synthetic':
    tre_list_np = np.array(tre_list)

    # sitk.WriteImage(
    #     get_moved_fixed_checkerboard(framework,
    #                                  moving_image_paths[tre_list_np.argmin()],
    #                                  fixed_image_paths[tre_list_np.argmin()]),
    #     os.path.join(output_path, 'tre_min_checkerboard.nrrd'))

    # sitk.WriteImage(
    #     get_moved_fixed_checkerboard(framework,
    #                                  moving_image_paths[tre_list_np.argmax()],
    #                                  fixed_image_paths[tre_list_np.argmax()]),
    #     os.path.join(output_path, 'tre_max_checkerboard.nrrd'))
    results_dict['tre_list'] = tre_list
    results_dict['tre_non_reg_list'] = tre_non_reg_list
    results_dict['tre_mean'] = tre_list_np.mean()
    results_dict['tre_non_reg_mean'] = np.array(tre_non_reg_list).mean()
    results_dict['tre_min'] = np.min(tre_list_np)
    results_dict['tre_max'] = np.max(tre_list_np)
    results_dict['tre_std'] = np.std(tre_list_np)
    results_dict['tre_25'] = np.percentile(tre_list_np, 25)
    results_dict['tre_50'] = np.percentile(tre_list_np, 50)
    results_dict['tre_75'] = np.percentile(tre_list_np, 75)
    results_dict['tre_var'] = np.var(tre_list_np)

json.dump(results_dict,
          open(os.path.join(output_path, 'resultsswaped.json'), 'w'),
          indent=4)
