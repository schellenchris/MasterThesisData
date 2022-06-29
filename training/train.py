import json
from DataHandler import DataHandler
#from trainPytorch import train_vxm_model
from TrainVoxelmorph import train_vxm_model

batch_size = 1
batch_size_val = 1
epochs = 1000
learning_rate = 0.00003
nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]
#nb_features = [[16, 16, 32, 32], [32, 32, 32, 32, 32, 16, 16]]
#nb_features = [[16, 16, 32, 32], [32, 32, 32, 16, 16]]
#nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 16, 16]]
#nb_features = [[16, 32, 32], [32, 32, 16]]
#nb_features =  [[8, 16, 16], [16, 16, 8]]
resampling = False
dataset = 'synthetic'
multi_gpu = True
dh = DataHandler()
if dataset == 'synthetic':
    dh.get_synthetic_data(
        fixed_path='/home/cschellenberger/Documents/newT3Resample1_8',
        moving_path='/home/cschellenberger/Documents/T1ResampledSpacing1_8')
    # dh.get_synthetic_data(
    #     fixed_path=
    #     '/home/cschellenberger/datam2olie/synthetic/orig/t3/Synthetic_CT/',
    #     moving_path=
    #     '/home/cschellenberger/datam2olie/synthetic/orig/t1/Synthetic_MR/')
    # dh.get_synthetic_data(
    #     fixed_path=
    #     '/home/cschellenberger/Documents/newT3v2/',
    #     moving_path=
    #     '/home/cschellenberger/datam2olie/synthetic/orig/t1/Synthetic_MR/')
    # dh.add_synthetic_data(
    #     fixed_path=
    #     '/home/cschellenberger/datam2olie/synthetic/orig/t1/Synthetic_CT/',
    #     moving_path=
    #     '/home/cschellenberger/datam2olie/synthetic/orig/t3/Synthetic_MR/')
elif dataset == 'L2R':
    dh.get_synthetic_data(
        fixed_path=
        '/home/cschellenberger/Documents/L2R_Resampled/L2R_Task1_CT/',
        moving_path=
        '/home/cschellenberger/Documents/L2R_Resampled/L2R_Task1_MR/')
elif dataset == 'mnist':
    dh.get_mnist_data(select_number=5)
else:
    raise NotImplementedError(
        f'{dataset} is not implemented yet please select one of the following losses [mnist, synthetic]'
    )

steps = len(dh.x_train) // batch_size
train_generator = dh.data_gen_voxelmorph(data_x=dh.x_train,
                                         data_y=dh.y_train,
                                         random_resampling=resampling,
                                         batch_size=batch_size,
                                         shuffle=True)
val_generator = dh.data_gen_voxelmorph(data_x=dh.x_val,
                                       data_y=dh.y_val,
                                       random_resampling=False,
                                       batch_size=batch_size_val,
                                       shuffle=False)

model_name = f'newT3_best_localmi_reg001_08_2nd_{epochs}_st{steps}_lr{str(learning_rate).replace(".", "_")}_bat{batch_size}{"withResampling" if resampling else ""}'

model_path = train_vxm_model(train_generator,
                             val_generator,
                             multi_gpu=multi_gpu,
                             steps_per_epoch=steps,
                             learning_rate=learning_rate,
                             loss='NMI',
                             model_name=model_name,
                             dataset=dataset,
                             epochs=epochs,
                             nb_features=nb_features,
                             int_resolution=1)