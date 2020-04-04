# Unpaired-GANCS

This code implements the recovery of image x from the undersampled measurements y when the pair (x,y) is not avaialble for training. However, we have a small amount of ground-truth x from y. This is important for medical imaging applications where usually one doesn't have access to high-resolution datastes for all organs. 

# Command line

python3 wgancs_main.py 

--dataset_train ./Knee-highresolution-19cases/train_small 

--dataset_label ./Knee-highresolution-19cases/partial_labels

--dataset_test ./Knee-highresolution-19cases/test_small 

--sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_5fold_160_128_knee_vdrad.mat 

--sample_size 320 --sample_size_y 256 

--batch_size 8 --sample_test 24 

--summary_period 1700 

--train_dir ./train_dir/exp29 

--checkpoint_dir ./checkpoint/exp29 

--wgan_gp True 

--activation lrelu 

--learning_rate_start 5e-5

# Datasets

For medical image reconstruction we adopt the MRI datasets available at the https://www.mridata.org in the "Stanford Fullysampled 3D FSE Knees" project, made available as a result of a joint collaboration between Stanford & UC Berkeley. It includes a 20 3D Knee images that have a high resoltuion of 320x320x256. 320 2D axial slices are collected from all patients to form the training and test datasets. 
