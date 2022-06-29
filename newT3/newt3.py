import SimpleITK as sitk
from DataHandler import DataHandler
import pickle
import numpy as np
import os
import pathlib

def findPixel(i, j, k, image):
    if k < image.GetSize()[2] - 1 and displacement.GetPixel(i, j, k + 1) != (0, 0, 0): return displacement.GetPixel(i, j, k + 1)
    if k > 0 and displacement.GetPixel(i, j, k - 1) != (0, 0, 0): return displacement.GetPixel(i, j, k - 1)
    return (0, 0, 0)

def get_landmarks(fixed_image_path: str, indexing: str = 'zyx', preResample = False):
    model_name = os.path.basename(fixed_image_path).replace('_atn_3.nrrd', '')
    if preResample: 
        if model_name == 'CT_Model151_Energy110': model_name = 'CT_Model151_Energy100'
        loaded_points = pickle.load(open(f'/home/cschellenberger/Documents/vectorPickles/{model_name}_vec_frame1_to_frame2.p', "rb"))
        moving_landmarks = [(float(loaded_points[idx]['1X']), float(loaded_points[idx]['1Y']), float(loaded_points[idx]['1Z'])) for (idx, _) in enumerate(loaded_points)]
        fixed_landmarks = [(float(loaded_points[idx]['2X']), float(loaded_points[idx]['2Y']), float(loaded_points[idx]['2Z'])) for (idx, _) in enumerate(loaded_points)]
    else: 
        loaded_points = pickle.load(open(f'/home/cschellenberger/Documents/vectorPickles/CT_points_t1_t3_withRegion_Continuous/{model_name}_idx.p', "rb"))
        moving_landmarks = loaded_points['t1']
        fixed_landmarks = loaded_points['t3']
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
    return moving_landmarks, fixed_landmarks, regions
    
dh = DataHandler(val_images=0)
dh.get_synthetic_data(
    fixed_path='/home/cschellenberger/datam2olie/synthetic/native/t3/Synthetic_CT/',
    moving_path='/home/cschellenberger/datam2olie/synthetic/native/t1/Synthetic_CT/')
moving_image_paths = dh.x_train
fixed_image_paths = dh.y_train
for idx in range(16, len(moving_image_paths)):
    print(idx)
    moving_image = sitk.ReadImage(moving_image_paths[idx])
    fixed_image = sitk.ReadImage(fixed_image_paths[idx])
    fixed_landmarks, moving_landmarks, regions = get_landmarks(fixed_image_paths[idx], indexing='xyz', preResample = True)
    resampler = sitk.ResampleImageFilter()
    displacement_np = np.array(fixed_landmarks) - np.array(moving_landmarks)
    displacement = sitk.Image(moving_image.GetSize(), sitk.sitkVectorFloat64)
    displacement.SetSpacing(moving_image.GetSpacing())
    displacement.SetOrigin(moving_image.GetOrigin())
    for i, point in enumerate(moving_landmarks):
        try: displacement.SetPixel(int(point[0]), int(point[1]), int(point[2]), displacement_np[i] * (1, 1, 2))
        except: continue
    for i in range(moving_image.GetSize()[0]):
        for j in range(moving_image.GetSize()[1]):
            for k in range(moving_image.GetSize()[2]):
                if displacement.GetPixel(i, j, k) == (0, 0, 0):
                    displacement.SetPixel(i, j, k, findPixel(i, j, k, moving_image))
    displacement_transform = displacement.__copy__()
    displacement_transform = sitk.DisplacementFieldTransform(displacement_transform)
    resampler.SetReferenceImage(moving_image)
    resampler.SetTransform(displacement_transform)
    newT3 = resampler.Execute(moving_image)
    for i in range(moving_image.GetSize()[0]):
        for j in range(moving_image.GetSize()[1]):
            for k in range(moving_image.GetSize()[2]):
                if -1 < newT3.GetPixel(i, j, k) < 1:
                    newT3.SetPixel(i, j, k, fixed_image.GetPixel(i, j, k))
    path = pathlib.Path(fixed_image_paths[idx])
    parts = list(path.parts)
    sitk.WriteImage(newT3, f'/home/cschellenberger/Documents/newT3preResample/{parts[-1]}')