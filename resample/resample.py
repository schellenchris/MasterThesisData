#%%
from pyM2aia import M2aiaOnlineHelper
import SimpleITK as sitk
import numpy as np
from typing import Tuple, Any

ctImg = sitk.ReadImage("/home/cschellenberger/datam2olie/Learn2Reg/Train/img0002_tcia_CT.nii")
mrImg = sitk.ReadImage("/home/cschellenberger/datam2olie/Learn2Reg/Train/img0002_tcia_MR.nii")
testImg = sitk.ReadImage("./MR_Model71_Energy100_act_1.nrrd")
ctTestImg = sitk.ReadImage("./CT_Model71_Energy100_atn_1.nrrd")
M2aiaHelper = M2aiaOnlineHelper("ipynbViewer", "jtfc.de:5050/m2aia/m2aia-no-vnc:with_exit", "8899")

#%%
# ctImg: sitk.Image = sitk.ScalarToRGBColormap(ctImg, sitk.ScalarToRGBColormapImageFilter.Jet)
# mrImg: sitk.Image = sitk.ScalarToRGBColormap(mrImg, sitk.ScalarToRGBColormapImageFilter.Jet)

# rgbImage = -rgbImage
minMaxFilter = sitk.MinimumMaximumImageFilter()
minMaxFilter.Execute(testImg)
maxPixel = minMaxFilter.GetMaximum() * 2
# for img in [ctImg, mrImg]:
#     minMaxFilter.Execute(img)
#     maxPixel = minMaxFilter.GetMaximum() * 2
#     for i in range(3):
#        for j in range(3):
#            img.SetPixel(256, 251 + i, 48 + j, maxPixel)
#            img.SetPixel(256, 254 + i, 48 + j, maxPixel)
#            img.SetPixel(256, 257 + i, 48 + j, maxPixel)

p = ctTestImg.TransformIndexToPhysicalPoint((188, 311, 63))
idx = testImg.TransformPhysicalPointToIndex(p)
for i in range(3):
    for j in range(3):
        for k in range(3):
            testImg.SetPixel(idx[0] + k, idx[1] + i, idx[2] + j, maxPixel)

print(idx)

sitk.WriteImage(testImg, "./MR_Model71_Energy100_act_1_marked.nrrd")
# sitk.WriteImage(mrImg, "./img0002_tcia_MR_marked.nrrd")

with M2aiaHelper as helper:
    helper.show({"nameOfImage0": ctImg, "nameOfImage1": ctTestImg, "test": testImg})


#%%
def resample_image(image, size, spacing, origin, interpolator=2) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size.tolist())
    resampler.SetOutputSpacing(spacing.tolist())
    resampler.SetOutputOrigin(origin.tolist())
    image_np = sitk.GetArrayFromImage(image)
    resampler.SetDefaultPixelValue(int(image_np.min()))
    resampler.SetInterpolator(interpolator)
    image_resampled = resampler.Execute(image)
    normalized_image = normalize_image(image_resampled)
    return normalized_image


def normalize_image(image: sitk.Image) -> sitk.Image:
    image_np = sitk.GetArrayFromImage(image)
    min_value = image_np.min()
    max_value = image_np.max()
    image_np = (image_np - min_value) / (max_value - min_value)
    image_result = sitk.GetImageFromArray(image_np)
    image_result.SetSpacing(image.GetSpacing())
    image_result.SetOrigin(image.GetOrigin())
    return image_result


def resample_image_0_0_0_centered(image_path: str, size: np.array,
                                  spacing: np.array) -> sitk.Image:
    image = sitk.ReadImage(image_path)
    # print(image.GetDirection())
    # image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    new_origin = calculate_origin(image)
    image.SetOrigin(new_origin.tolist())
    new_origin = calculate_origin(image, size, spacing)
    image_resampled = resample_image(image, size, spacing, new_origin)
    return image_resampled


def calculate_origin(image: sitk.Image, size=None, spacing=None) -> np.array:
    if size is None or spacing is None:
        size = np.array(image.GetSize())
        spacing = np.array(image.GetSpacing())
        direction = np.array(image.GetDirection()).reshape((3, 3))
        return -size * (direction @ spacing) / 2.0
    else:
        return -size * spacing / 2.0


all_image_paths = ["./img0002_tcia_CT_marked.nii", "./img0002_tcia_MR_marked.nii",
                   "./CT_Model71_Energy100_atn_1_marked.nrrd", "./CT_Model71_Energy90_atn_3.nrrd",
                   "./MR_Model71_Energy100_act_1_marked.nrrd", "./MR_Model71_Energy100_act_3.nrrd"]
new_size, new_spacing = np.array((256, 256, 128)), np.array((1.8, 1.8, 1.8))
for path in all_image_paths:
    image = sitk.ReadImage(path)
    image_resampled = resample_image_0_0_0_centered(path, new_size, new_spacing)
    sitk.WriteImage(image_resampled, path[:-4] + "_resampled.nrrd")

#%%
origCtImg = sitk.ReadImage("./img0002_tcia_CT_marked.nii")
ctImg = sitk.ReadImage("./img0002_tcia_CT_marked_resampled.nrrd")
mrImg = sitk.ReadImage("./img0002_tcia_MR_marked_resampled.nrrd")
testImg = sitk.ReadImage("./resampleTest.nrrd")
testImgResampled = sitk.ReadImage("./CT_Model71_Energy90_atn_3_marked._resampled.nrrd")
segCT = sitk.ReadImage("./seg_01_logM_iso_CT.nii")
segMR = sitk.ReadImage("./seg_1_logM_chaos_MR.nii")

M2aiaHelper = M2aiaOnlineHelper("ipynbViewer", "jtfc.de:5050/m2aia/m2aia-no-vnc:with_exit", "8899")
with M2aiaHelper as helper:
    helper.show({"ctImg": ctImg, "mrImg": mrImg, "origCtImg": origCtImg, "testImg": testImg,
                 "testImgResampled": testImgResampled, "segCT": segCT, 'segMR': segMR})

#%%
def calculate_shift(image: sitk.Image) -> np.array:
    center_mm = (np.array(image.GetSize()) * np.array(image.GetSpacing())) / 2
    shift_vector_mm = center_mm * (-1)
    return shift_vector_mm


def get_idx_resampled(image_native: sitk.Image, image_resampled: sitk.Image,
                      shift_vector_mm: np.array,
                      point: tuple) -> Tuple[Any, Any]:
    point_native_mm = image_native.TransformContinuousIndexToPhysicalPoint(
        point)
    point_resampled_mm = point_native_mm + shift_vector_mm
    point_resampled_idx = image_resampled.TransformPhysicalPointToIndex(
        point_resampled_mm)
    return (point_resampled_idx, point_resampled_mm)


t1_native = sitk.ReadImage("./CT_Model71_Energy100_atn_1_marked.nrrd")
t1_resampled = sitk.ReadImage("./MR_Model71_Energy100_act_1_marked._resampled.nrrd")
t3_native = sitk.ReadImage("./CT_Model71_Energy90_atn_3.nrrd")
t3_resampled = sitk.ReadImage("./MR_Model71_Energy100_act_3._resampled.nrrd")
size_resampled = t1_resampled.GetSize()
vector_field_path = "./CT_Model71_Energy90_vec_frame1_to_frame2.txt"

shift_vector_t1_mm = calculate_shift(t1_native)
shift_vector_t3_mm = calculate_shift(t3_native)
vector_field = np.genfromtxt(vector_field_path,
                             usecols=(2, 3, 4, 6, 7, 8),
                             names='1X, 1Y, 1Z, 2X, 2Y, 2Z',
                             dtype=None,
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

for (point_idx, _) in enumerate(points_t1):
    point_t1_resampled_idx, point_t1_resampled_mm = get_idx_resampled(
        t1_native, t1_resampled, shift_vector_t1_mm, points_t1[point_idx])

    point_t3_resampled_idx, point_t3_resampled_mm = get_idx_resampled(
        t3_native, t3_resampled, shift_vector_t3_mm, points_t3[point_idx])

    # if all((y >= t1 >= 0) for t1, y in zip(point_t1_resampled_idx, size_resampled)
    #     ) and t1_resampled.GetPixel(point_t1_resampled_idx) != 0.0:
    points_t1_resampled_idx.append(point_t1_resampled_idx)
    points_t1_resampled_mm.append(point_t1_resampled_mm)
    points_t3_resampled_idx.append(point_t3_resampled_idx)
    points_t3_resampled_mm.append(point_t3_resampled_mm)

#%%
idx = points_t1.index((188, 311, 63))
print(points_t1_resampled_idx[idx])
print(points_t1_resampled_mm[idx])
t3_idx = points_t3_resampled_idx[idx]
print(t3_idx)
print(points_t3_resampled_mm[idx])

t3Resampled = sitk.ReadImage("./MR_Model71_Energy100_act_3._resampled.nrrd")
minMaxFilter = sitk.MinimumMaximumImageFilter()
minMaxFilter.Execute(t3Resampled)
maxPixel = minMaxFilter.GetMaximum() * 2
for i in range(3):
    for j in range(3):
        for k in range(3):
            t3Resampled.SetPixel(t3_idx[0] + k, t3_idx[1] + i, t3_idx[2] + j, maxPixel)

sitk.WriteImage(t3Resampled, "./MR_Model71_Energy100_act_3_marked._resampled.nrrd")

#%%
t1Resampled = sitk.ReadImage("./CT_Model71_Energy100_atn_1_marked._resampled.nrrd")
t3Resampled = sitk.ReadImage("./CT_Model71_Energy90_atn_3_marked._resampled.nrrd")
t1 = sitk.ReadImage("./CT_Model71_Energy100_atn_1.nrrd")
t3 = sitk.ReadImage("./CT_Model71_Energy90_atn_3_marked.nrrd")

M2aiaHelper = M2aiaOnlineHelper("ipynbViewer", "jtfc.de:5050/m2aia/m2aia-no-vnc:with_exit", "8899")
with M2aiaHelper as helper:
    helper.show({"t1Resampled": t1Resampled, "t3Resampled": t3Resampled, "t1": t1, "t3": t3})

#%%
t1Resampled = sitk.ReadImage("./MR_Model71_Energy100_act_1_marked._resampled.nrrd")
t3Resampled = sitk.ReadImage("./MR_Model71_Energy100_act_3_marked._resampled.nrrd")
t1ResampledCT = sitk.ReadImage("./CT_Model71_Energy100_atn_1_marked._resampled.nrrd")
t3ResampledCT = sitk.ReadImage("./CT_Model71_Energy90_atn_3_marked._resampled.nrrd")
t1 = sitk.ReadImage("./MR_Model71_Energy100_act_1_marked.nrrd")
t3 = sitk.ReadImage("./MR_Model71_Energy100_act_3.nrrd")

M2aiaHelper = M2aiaOnlineHelper("ipynbViewer", "jtfc.de:5050/m2aia/m2aia-no-vnc:with_exit", "8899")
with M2aiaHelper as helper:
    helper.show(
        {"t1Resampled": t1Resampled, "t3Resampled": t3Resampled, "t1": t1, "t3": t3, "t1ResampledCT": t1ResampledCT,
         "t3ResampledCT": t3ResampledCT})