import SimpleITK as sitk
from DataHandler import DataHandler
import pickle
import numpy as np
import os

def get_moved_points(points: np.array, displacement: sitk.Image) -> np.array:
    displacement_copy = displacement.__copy__()
    displacement_transform = sitk.DisplacementFieldTransform(displacement_copy)
    moved_points = [displacement_transform.TransformPoint(point) for point in points]
    return moved_points

def get_landmarks(fixed_image_path: str, indexing: str = 'zyx', continuous = False, preResample = False):
    model_name = os.path.basename(fixed_image_path).replace('_atn_3.nrrd', '')
    if preResample: 
        loaded_points = pickle.load(open(f'/home/cschellenberger/Documents/vectorPickles/{model_name}_vec_frame1_to_frame2.p', "rb"))
        moving_landmarks = np.array([(float(loaded_points[idx]['1X']), float(loaded_points[idx]['1Y']), float(loaded_points[idx]['1Z'])) for (idx, _) in enumerate(loaded_points)])
        fixed_landmarks = np.array([(float(loaded_points[idx]['2X']), float(loaded_points[idx]['2Y']), float(loaded_points[idx]['2Z'])) for (idx, _) in enumerate(loaded_points)])
    else: 
        if continuous: loaded_points = pickle.load(open(f'/home/cschellenberger/Documents/vectorPickles/CT_points_t1_t3_withRegion_Spacing1_8/{model_name}_idx.p', "rb"))
        #if continuous: loaded_points = pickle.load(open(f'/home/cschellenberger/Documents/vectorPickles/CT_points_t1_t3_withRegion_Continuous/{model_name}_idx.p', "rb"))
        else: loaded_points = pickle.load(open(f'/home/cschellenberger/Documents/vectorPickles/CT_points_t1_t3_withRegion/{model_name}_idx.p', "rb"))
        moving_landmarks = np.array(loaded_points['t1'])
        fixed_landmarks = np.array(loaded_points['t3'])
    regions = np.array(loaded_points['Region'])
    if indexing == 'xzy':
        # swap columns because numpy and vxm use zyx indexing and the data uses xyz indexing
        moving_landmarks[:, [1, 2]] = moving_landmarks[:, [2, 1]]
        fixed_landmarks[:, [1, 2]] = fixed_landmarks[:, [2, 1]]
    elif indexing == 'zyx':
        # swap columns because numpy and vxm use zyx indexing and the data uses xyz indexing
        moving_landmarks[:, [0, 2]] = moving_landmarks[:, [2, 0]]
        fixed_landmarks[:, [0, 2]] = fixed_landmarks[:, [2, 0]]
    elif indexing == 'yxz':
        # swap columns because numpy and vxm use zyx indexing and the data uses xyz indexing
        moving_landmarks[:, [0, 1]] = moving_landmarks[:, [1, 0]]
        fixed_landmarks[:, [0, 1]] = fixed_landmarks[:, [1, 0]]
    else: assert indexing == 'xyz', f'indexing can only be xyz or zyx. Got: {indexing}'
    return moving_landmarks.astype(np.float64), fixed_landmarks.astype(np.float64), regions
    
idx = 0
dh = DataHandler(val_images=12)
# dh.get_synthetic_data(
#     fixed_path='/home/cschellenberger/datam2olie/synthetic/orig/t3/Synthetic_CT/',
#     moving_path='/home/cschellenberger/datam2olie/synthetic/orig/t1/Synthetic_MR/')
dh.get_synthetic_data(
    fixed_path='/home/cschellenberger/Documents/newT3Resample1_8',
    moving_path='/home/cschellenberger/Documents/T1ResampledSpacing1_8')
moving_image_paths = dh.x_val
fixed_image_paths = dh.y_val
resErr = 0
dists = {}
for idx in range(len(moving_image_paths)):
    moving_image = sitk.ReadImage(moving_image_paths[idx])
    fixed_image = sitk.ReadImage(fixed_image_paths[idx])
    elastix_image_filter = sitk.ElastixImageFilter()
    elastix_image_filter.SetFixedImage(fixed_image)
    elastix_image_filter.SetMovingImage(moving_image)
    parameterMap0 = sitk.ReadParameterFile('./params.txt')
    elastix_image_filter.SetParameterMap(parameterMap0)
    moving_landmarks, fixed_landmarks, regions = get_landmarks(fixed_image_paths[idx], indexing='xyz', continuous = True)
    elastix_image_filter.Execute()
    transformixFilter = sitk.TransformixImageFilter()
    transformixFilter.SetTransformParameterMap(elastix_image_filter.GetTransformParameterMap())
    transformixFilter.ComputeDeformationFieldOn()
    transformixFilter.SetMovingImage(moving_image)
    transformixFilter.Execute()
    moved = transformixFilter.GetResultImage()
    displacement = sitk.GetImageFromArray(sitk.GetArrayFromImage(transformixFilter.GetDeformationField()) / 1.8) 
    displacement = sitk.Cast(displacement, sitk.sitkVectorFloat64)
    moved_landmarks = get_moved_points(fixed_landmarks, displacement)
    err = moving_landmarks - moved_landmarks
    err = np.linalg.norm(err, axis = 1)
    resErr += err.mean()
    dists[f'dist{idx}'] = err.mean()
print(resErr / len(moving_image_paths))
pickle.dump(dists, open("./ElastixResults/GridSpacing60NewT3.p", "wb"))