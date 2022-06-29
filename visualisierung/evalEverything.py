import numpy as np
import pickle
from TrainVoxelmorph import VoxelmorphTF
from DataHandler import DataHandler
import SimpleITK as sitk
import os
from DataHandler import DataHandler
import pickle
import h5py

def getLayerSize(weight_file_path):
    f = h5py.File(weight_file_path)
    dec, enc = [], []
    try:
        for layer, g in f.items():
            for name, d in g.items(): 
                if str(f.filename).endswith('.h5'):
                    for k, v in d.items():
                        if 'kernel' in k: dec.append(np.array(v).shape[4]) if 'dec' in name else enc.append(np.array(v).shape[4])
    finally:
        f.close()
    dec.sort(reverse=True)
    return [enc[1:], dec]

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

for path in ['Learningrate']: #os.listdir('/home/cschellenberger/Documents/scripts/models/thesis')
    print(path)
    for model in os.listdir('/home/cschellenberger/Documents/scripts/models/thesis/' + path):
        if model == 'Results': continue
        weights_path = '/home/cschellenberger/Documents/scripts/models/thesis/' + path + '/' + model + '/bestWeights.h5'
        downsize = int(model[0]) if path == 'Downsample' else 1
        dh = DataHandler(val_images=12)
        if 'newT3' in model:
            dh.get_synthetic_data(
                fixed_path='/home/cschellenberger/Documents/newT3Resample1_8',
                moving_path='/home/cschellenberger/Documents/T1ResampledSpacing1_8')
        else:
            dh.get_synthetic_data(
                fixed_path='/home/cschellenberger/datam2olie/synthetic/orig/t3/Synthetic_CT/',
                moving_path='/home/cschellenberger/datam2olie/synthetic/orig/t1/Synthetic_MR/')
        dists = {}
        moving_image_paths = dh.x_val
        fixed_image_paths = dh.y_val
        nb_features = getLayerSize(weights_path)
        device = '/cpu:0'
        imgReg = VoxelmorphTF(weights_path, sitk.ReadImage(fixed_image_paths[0]), nb_features, downsize)
        for i in range(len(fixed_image_paths)):
            fixed_image = sitk.ReadImage(fixed_image_paths[i])
            moving_image = sitk.ReadImage(moving_image_paths[i])
            moving_landmarks, fixed_landmarks, regions = get_landmarks(fixed_image_paths[i], indexing='xyz', continuous=True)
            disp_np = moving_landmarks - fixed_landmarks
            disp_np = np.linalg.norm(disp_np, axis = 1)
            disp_image = sitk.Image(fixed_image.GetSize(), sitk.sitkFloat64)
            disp_image.SetOrigin(fixed_image.GetOrigin())
            disp_image.SetSpacing(fixed_image.GetSpacing())
            for idx, point in enumerate(fixed_landmarks):
                try: disp_image.SetPixel(int(point[0]), int(point[1]), int(point[2]), disp_np[idx])
                except: continue
            moved_img, displacement_np, time = imgReg.register_images(moving_image, fixed_image, device)
            displacement = displacement_np.squeeze()
            displacement[:,:,:,[0,2]] = displacement[:,:,:,[2,0]]
            displacement = sitk.GetImageFromArray(displacement.astype(np.float64), isVector=True)
            displacement.SetSpacing((downsize, downsize, downsize))
            moved_landmarks = get_moved_points(fixed_landmarks, displacement)
            err = moving_landmarks - moved_landmarks
            err = np.linalg.norm(err, axis = 1)
            err_image = sitk.Image(fixed_image.GetSize(), sitk.sitkFloat64)
            err_image.SetSpacing(fixed_image.GetSpacing())
            err_image.SetOrigin(fixed_image.GetOrigin())
            for idx, point in enumerate(fixed_landmarks):
                try: err_image.SetPixel(int(point[0]), int(point[1]), int(point[2]), err[idx])
                except: continue
            dists[f'dist{i}'] = err.mean()
        pickle.dump(dists, open('/home/cschellenberger/Documents/scripts/models/thesis/' + path + '/Results/' + model + '.p', "wb"))
                