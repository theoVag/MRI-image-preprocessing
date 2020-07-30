# MRI-image-preprocessing
This repository includes the preprocessing steps for MRI Knee images. These files are then imported in my main model for Image Registration and Segmentation using Deep Learning. This project is the initial pipeline process applied to a part of Osteoarthritis Initiative dataset, in order to prepare the MRI images to train and test my framework. The preprocessing and the main model (which will be uploaded in the following weeks) are parts of the source code implemented for my Msc thesis with title "Deep learning neural networks for the registration and cartilage segmentation of MRI knee images" in Advanced Computer and Communication Systems with specialization field: Intelligent and Autonomous Systems-Computational Intelligence Methodologies and Applications.

## Execution options and script
pipeline_preprocess.py: is a script for serial execution of all basic preprocessing steps applied to the Osteoarthritis Initiative dataset. It needs a path for the folder that contains the part of the dataset.
preprocessSingle.py: is a parameterized script that executes the basic steps only for a single test image and plots the results for verification.

## Files description
FullDataPreparation.py : Class for reading and preparing dicom and mhd files in order to save them in the required npz form.
ImgProcess.py :  Class for affine aligning of all images in the Baseline folder.
img_utils.py : Useful functions and their wrappers for image preprocessing like resampling, denoising, compressing
normImages.py : Normalization functions and required wrappers
plots_lib.py : Plotting functions for image preprocess analysis and verification
DataSplit.py : Class which contains all help functions for splitting dataset into train/validation/test according to the parameters (size, filename list, stratified by Kellgren-Lawrence)
