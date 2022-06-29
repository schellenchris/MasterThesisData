import os

import SimpleITK as sitk
import numpy as np
from DataHandler import DataHandler
import pickle
import pandas as pd

def calculate_origin(image: sitk.Image, size=None, spacing=None) -> np.array:
    if size is None or spacing is None:
        size = np.array(image.GetSize())
        spacing = np.array(image.GetSpacing())
        direction = np.array(image.GetDirection()).reshape((3, 3))
        return -size * (direction @ spacing) / 2.0
    else:
        return -size * spacing / 2.0


def calculate_shift(image: sitk.Image) -> np.array:
    center_mm = (np.array(image.GetSize()) * np.array(image.GetSpacing())) / 2
    shift_vector_mm = center_mm * (-1)
    return shift_vector_mm


def get_idx_resampled(image_native: sitk.Image, image_resampled: sitk.Image,
                      shift_vector_mm: np.array,
                      point: tuple) -> (tuple, tuple):
    point_native_mm = image_native.TransformContinuousIndexToPhysicalPoint(
        point)
    point_resampled_mm = point_native_mm + shift_vector_mm
    point_resampled_idx = image_resampled.TransformPhysicalPointToContinuousIndex(
        point_resampled_mm)
    return point_resampled_idx, point_resampled_mm


dh = DataHandler(val_images=12)
# dh.get_synthetic_data(
#     fixed_path='/home/cschellenberger/datam2olie/synthetic/orig/t3/Synthetic_CT/',
#     moving_path='/home/cschellenberger/datam2olie/synthetic/orig/t1/Synthetic_CT/',
#     traverse_sub_dir=False)
dh.get_synthetic_data(
    fixed_path='/home/cschellenberger/Documents/newT3Resampled',
    moving_path='/home/cschellenberger/Documents/T1ResampledSpacing2',
    traverse_sub_dir=False)
t1s_resampled = dh.x_val
t3s_resampled = dh.y_val
dh.get_synthetic_data(
    fixed_path='/home/cschellenberger/datam2olie/synthetic/native/t3/Synthetic_CT/',
    moving_path='/home/cschellenberger/datam2olie/synthetic/native/t1/Synthetic_CT/',
    traverse_sub_dir=False)
t1s_native = dh.x_val
t3s_native = dh.y_val

output_dir = '/home/cschellenberger/Documents/vectorPickles/CT_points_t1_t3_withRegion/'
vector_fields_dir = '/home/cschellenberger/datam2olie/synthetic/native/CT_vector_fields/'

for (image_idx, _) in enumerate(t3s_resampled):
    print(image_idx)
    t1_native = sitk.ReadImage(t1s_native[image_idx])
    t3_native = sitk.ReadImage(t3s_native[image_idx])
    t1_resampled = sitk.ReadImage(t1s_resampled[image_idx])
    t3_resampled = sitk.ReadImage(t3s_resampled[image_idx])
    size_resampled = t1_resampled.GetSize()
    model_name = os.path.basename(t3s_native[image_idx]).replace(
        '_atn_3.nrrd', '')
    vector_field_path = f'{vector_fields_dir}{model_name}_vec_frame1_to_frame2.txt'

    shift_vector_t1_mm = calculate_shift(t1_native)
    shift_vector_t3_mm = calculate_shift(t3_native)

    vector_field = np.genfromtxt(vector_field_path,
                                 usecols=(0, 2, 3, 4, 6, 7, 8),
                                 names='Region, 1X, 1Y, 1Z, 2X, 2Y, 2Z',
                                 dtype=('<U34', float, float, float, float, float, float),
                                 skip_header=2)

    points_t1 = [
        (float(vector_field[idx]['1X']), float(vector_field[idx]['1Y']),
         float(vector_field[idx]['1Z']))
        for (idx, _) in enumerate(vector_field)
    ]
    points_t3 = [
        (float(vector_field[idx]['2X']), float(vector_field[idx]['2Y']),
         float(vector_field[idx]['2Z']))
        for (idx, _) in enumerate(vector_field)
    ]

    points_t1_resampled_idx = []
    points_t3_resampled_idx = []
    points_t1_resampled_mm = []
    points_t3_resampled_mm = []
    regions = []

    for (point_idx, _) in enumerate(points_t1):
        point_t1_resampled_idx, point_t1_resampled_mm = get_idx_resampled(
            t1_native, t1_resampled, shift_vector_t1_mm, points_t1[point_idx])
        point_t3_resampled_idx, point_t3_resampled_mm = get_idx_resampled(
            t3_native, t3_resampled, shift_vector_t3_mm, points_t3[point_idx])

        try:
            px = t1_resampled.GetPixel(point_t1_resampled_idx)
            points_t1_resampled_idx.append(point_t1_resampled_idx)
            points_t3_resampled_idx.append(point_t3_resampled_idx)
            points_t1_resampled_mm.append(point_t1_resampled_mm)
            points_t3_resampled_mm.append(point_t3_resampled_mm)
            regions.append(vector_field[point_idx]['Region'])
        except: continue
    residx = {'Region': regions, 't1': points_t1_resampled_idx, 't3': points_t3_resampled_idx}
    resmm = {'Region': regions, 't1': points_t1_resampled_mm, 't3': points_t3_resampled_mm}
    pickle.dump(residx, open(f"/home/cschellenberger/Documents/vectorPickles/CT_points_t1_t3_withRegion_Spacing1_8/{model_name}_idx.p", "wb"))
    pickle.dump(resmm, open(f"/home/cschellenberger/Documents/vectorPickles/CT_points_t1_t3_withRegion_Spacing1_8/{model_name}_mm.p", "wb"))
