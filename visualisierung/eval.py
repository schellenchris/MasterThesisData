import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from TrainVoxelmorph import VoxelmorphTF
from DataHandler import DataHandler
import SimpleITK as sitk

framework = VoxelmorphTF()
dh = DataHandler(val_images=12)
dh.get_synthetic_data(
    fixed_path='/home/cschellenberger/datam2olie/synthetic/orig/t3/Synthetic_CT/',
    moving_path='/home/cschellenberger/datam2olie/synthetic/orig/t1/Synthetic_MR/',
    traverse_sub_dir=False)
moving_image_paths = dh.x_val
fixed_image_paths = dh.y_val
moving_image = sitk.ReadImage(moving_image_paths[0])
fixed_image = sitk.ReadImage(fixed_image_paths[0])
print(fixed_image_paths[0])
moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
moved_image, displacement, time = framework.register_images(
        fixed_image=fixed_image,
        moving_image=moving_image,
        weights_path=model_path,
        loss=loss)
vector_field_path = "/home/cschellenberger/Documents/vectorPickles/CT_Model106_Energy110_vec_frame1_to_frame2.p"
picklePaths = '/home/cschellenberger/Documents/vectorPickles/'
vector_field = pickle.load(open(vector_field_path, "rb"))

moving_landmarks, fixed_landmarks = get_landmarks(
            fixed_image_paths[0], indexing='xyz')
moved_landmarks = get_moved_points(fixed_landmarks, displacement)



p1 = np.array([df['1X'], df['1Y'], df['1Z']])
p2 = np.array([df['evalX'], df['evalY'], df['evalZ']])
squared_dist = np.sum((p1 - p2) ** 2, axis=0)
df['dist'] = np.sqrt(squared_dist)
dists = df.groupby(df['Region'])['dist'].sum() / df['Region'].value_counts()
dists = dists.sort_values(ascending=False)
ax = sns.barplot(y = df['Region'], x = df['dist'], order = dists.index[:10], errwidth = 0)
plt.tight_layout()
plt.show()
