#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:13:47 2019

@author: Theodoros- Panagiotis Vagenas
"""

# Functions and wrapper functions for image preprocessing and preparing

from glob import glob
import numpy as np
from time import time
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
import SimpleITK as sitk
import os
from skimage import exposure


# Median filter with rad
def median_filter(inputImage,rad=None):
    
    inputImage = sitk.GetImageFromArray(inputImage)
    medianFilter = sitk.MedianImageFilter()
    if rad is not None:
        medianFilter.SetRadius (rad)
    output = medianFilter.Execute(inputImage)
    output = sitk.GetArrayFromImage(output)
    #output = ndimage.median_filter(inputImage, size=3)
    #output = signal.medfilt(inputImage, 3)
    return output

# Wrapper to apply function in Baseline folder
def apply_median_filter(baseline_path,radius=None): # slices x d1 x d2
    path = baseline_path
    counter =0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("_test_vol.npz"):
                 print("\n Applying median filter to file #%d" % counter)           
                 vol_array = np.load(os.path.join(root, file))['vol_data']
                 volume_image = median_filter(vol_array,radius)
                 np.savez(os.path.join(root, file), vol_data=volume_image)
                 counter = counter + 1
                 
# Wrapper to apply function in Baseline folder
def apply_reduce_slices(baseline_path,left_size,right_size): # slices x d1 x d2
    path = baseline_path
    counter =0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("_test_vol.npz"):
                 print("\n Resampling file #%d" % counter)
                 seg_file = os.path.join(root, file).replace('_test_vol.npz','.segmentation_masks_test_seg.npz')
                 
                 vol_array = np.load(os.path.join(root, file))['vol_data']
                 seg_array = np.load(seg_file)['vol_data']
                 volume_image = reduce_slices(vol_array,left_size,right_size)
                 seg_image = reduce_slices(seg_array,left_size,right_size)
                 print(volume_image.shape)
                 np.savez(os.path.join(root, file), vol_data=volume_image)
                 np.savez(seg_file, vol_data=seg_image)
                 counter = counter + 1

# Remove slices from left and right
def reduce_slices(volume,left_size,right_size):
    
    ns = volume.shape
    volume = volume[left_size:(ns[0]-right_size),:,:]
    return volume

# Resize volume and his segmentation map to the target size    
def resize_sitk(volume_image_np,seg_volume_image_np,ns,isLabel=False):
        #new_size = [112,96,160]
        
        #num_slices = volume_image_np.shape[0]
        
        nsize = [ns[2],ns[1],ns[0]]
        image = sitk.GetImageFromArray(volume_image_np)
        labeled_image = sitk.GetImageFromArray(seg_volume_image_np)
        resample = sitk.ResampleImageFilter()
        if isLabel:
            
            resample.SetInterpolator = sitk.sitkNearestNeighbor
        else:
            
            resample.SetInterpolator = sitk.sitkLinear
        
        orig_spacing = image.GetSpacing()
        orig_size = np.array(image.GetSize(), dtype=np.int)
        new_spacing = (orig_size/nsize)*orig_spacing
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(nsize)
        resampled_image = resample.Execute(image)
        
        new_image = sitk.GetArrayFromImage(resampled_image)
        
        resample2 = sitk.ResampleImageFilter()
        orig_spacing = labeled_image.GetSpacing()
        orig_size = np.array(labeled_image.GetSize(), dtype=np.int)
        new_spacing = (orig_size/nsize)*orig_spacing
        resample2.SetOutputSpacing(new_spacing)
        resample2.SetSize(nsize)
        
        resample2.SetInterpolator(sitk.sitkNearestNeighbor)
        
        resampled_label_image = resample2.Execute(labeled_image)
        
        resampled_label_image = sitk.GetArrayFromImage(resampled_label_image)
        print(np.unique(resampled_label_image))
        # resample labels
        """tr = resample.GetTransform()
        label_resample = sitk.ResampleImageFilter()
        label_resample.SetInterpolator = sitk.sitkNearestNeighbor
        label_resample.SetTransform(tr)
        label_resample.SetOutputSpacing(new_spacing)
        label_resample.SetSize(nsize)"""
        
        """
        
        resampled_label_image = sitk.GetArrayFromImage(si.Execute(image))
        new_image = sitk.GetArrayFromImage(si.Execute(labeled_image))"""
        
        #print(new_image.shape)
        #num_slices = new_image.shape[0]
        """if isLabel:
            for i in range(num_slices-1):
                new_image[i,:,:] = np.flipud(new_image[i,:,:])"""
        
        
        return new_image,resampled_label_image

# Wrapper for function resize for Baseline folder
def apply_img_resample_Base(baseline_path="/home/tpvagenas/main_preprocess/Baseline_new",ns=[140,192,224]):
    
    path = baseline_path
    counter =0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("_test_vol.npz"):
                 print("\n Resampling file #%d" % counter)
                 seg_file = os.path.join(root, file).replace('_test_vol.npz','.segmentation_masks_test_seg.npz')
                 vol_array = np.load(os.path.join(root, file))['vol_data']
                 seg_array = np.load(seg_file)['vol_data']             
                 [volume_image,seg_image]=resize_sitk(vol_array,seg_array,ns)
                 
                 np.savez(os.path.join(root, file), vol_data=volume_image)
                 np.savez(seg_file, vol_data=seg_image)
                 counter = counter + 1

# Function for denoising                
def img_denoise(data, patch_rad = 3,block_rad = 4,n_sigma =4):
    
    #data = np.load(img)['vol_data']
    new_data = np.zeros((data.shape[1],data.shape[2],data.shape[0]))
    for ch in range(data.shape[0]):
        new_data[:,:,ch]=data[ch,:,:]
        
    sigma = estimate_sigma(new_data, N=n_sigma)
    #den = nlmeans(data, sigma=sigma, mask=mask, patch_radius= 1, block_radius = 1, rician= True)
    den = nlmeans(new_data, sigma=sigma, patch_radius= patch_rad, block_radius = block_rad, rician= True)
    #print("total time", time() - t)
    res_data = np.zeros(data.shape)
    for ch in range(data.shape[0]):
        res_data[ch,:,:]=den[:,:,ch]
    
    return res_data
    #np.savez(img,vol_data = res_data)

# Wrapper for function denoise for folder data
def apply_img_denoise_data(path_data="/home/tpvagenas/main_preprocess/data_corrected"):
    path = path_data
    train_files=glob(path+"/train/vols/*")
    validate_files=glob(path+"/validate/vols/*")
    test_files=glob(path+"/test/vols/*")
    
    print("Denoising train files...")
    for filename in train_files:
        print(filename)
        data = np.load(filename)['vol_data']
        res_data=img_denoise(data)
        np.savez(filename,vol_data = res_data)
    
    print("Denoising validate files...")
    for filename in validate_files:
        data = np.load(filename)['vol_data']
        res_data=img_denoise(data)
        np.savez(filename,vol_data = res_data)
    print("Denoising test files...")
    for filename in test_files:
        data = np.load(filename)['vol_data']
        res_data=img_denoise(data) 
        np.savez(filename,vol_data = res_data)

# Wrapper for function denoise for folder Baseline
def apply_img_denoise_Base(baseline_path="/home/tpvagenas/main_preprocess/Baseline_recon"):
    
    path = baseline_path
    counter =0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("_test_vol.npz"):
                 print("\n Denoising file #%d" % counter)
                 data = np.load(os.path.join(root, file))['vol_data']
                 res_data = img_denoise(data)
                 np.savez(os.path.join(root, file),vol_data = res_data)
                 counter = counter + 1
                 

# Function for Histogra matching       
def histogram_matching(image,reference,ns=[160,192,224]):
    ns = reference.shape # ns auto sto reshape
    #ns = image.shape # ns auto thelei sto reshape
    fixed = sitk.GetImageFromArray(reference)
    moving = sitk.GetImageFromArray(image)
    
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    matcher = sitk.HistogramMatchingImageFilter()
    if ( fixed.GetPixelID() in ( sitk.sitkUInt8, sitk.sitkInt8 ) ):
        matcher.SetNumberOfHistogramLevels(128)
    else:
        matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    moving = matcher.Execute(moving,fixed)
    moving = sitk.GetArrayFromImage(moving).reshape(ns)
    return moving

# Wrapper for function to folder Baseline
def apply_histogram_matching_Base(atlas_path,path="/home/tpvagenas/main_preprocess/Baseline_recon",ns=[160,192,224]):
    counter =0
    atlas_path=atlas_path +"_test_vol.npz"
    atlas = np.load(atlas_path)['vol_data']

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("_test_vol.npz"):
                 print("\n Histogram Matching file #%d" % counter)                       
                 volume_image = np.load(os.path.join(root, file))['vol_data']
                 volume_image = histogram_matching(volume_image,atlas,ns)
                 np.savez(os.path.join(root, file), vol_data=volume_image)
                 counter = counter + 1
    
# Function for constrast enhancement- stretching
def constrast_enhancement(image):
    # Contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    rescaled = exposure.rescale_intensity(image, in_range=(p2, p98))
    #rescaled = exposure.adjust_sigmoid(image, cutoff=0.1, gain=10, inv=False)
    
    return rescaled

# Wrapper for function for folder Baseline
def apply_constrast_enhancement_Base(path="/home/tpvagenas/main_preprocess/Baseline_recon"):
    counter =0

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("_test_vol.npz"):
                 print("\n Constrast stretching file #%d" % counter)                       
                 volume_image = np.load(os.path.join(root, file))['vol_data']
                 volume_image = constrast_enhancement(volume_image)
                 np.savez(os.path.join(root, file), vol_data=volume_image)
                 counter = counter + 1

# Function for normalization to 0-1
def normImages0_1(img,ns=[160,192,224]):
    
    img=sitk.GetImageFromArray(img)
    
    rescalFilt=sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(1)
    rescalFilt.SetOutputMinimum(0)
     
    img=rescalFilt.Execute(img)
    img=sitk.GetArrayFromImage(img).reshape(ns)
    
    return img

# Wrapper for function for folder Baseline
def normIm_Base(path,ns=[160,192,224]):

    print("Normalizing Baseline files")
    counter =1
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("_test_vol.npz"):
                 print("\n Normalizing file #%d" % counter)
                 print(os.path.join(root, file))
                 volume_image = np.load(os.path.join(root, file))['vol_data']
                 volume_image = normImages0_1(volume_image,ns)
                 np.savez(os.path.join(root, file), vol_data=volume_image)
                 counter = counter + 1
                 
def binary_mask(seg_image):
    
    new_seg_image = np.zeros(seg_image.shape)
    new_seg_image[seg_image==2] = 1
    new_seg_image[seg_image==4] = 1
    return new_seg_image
    

def affine_align(moving_image,moving_seg_image,fixed_image):
        
        moving_image=sitk.GetImageFromArray(moving_image)
        fixed_image=sitk.GetImageFromArray(fixed_image)
        moving_seg_image=sitk.GetImageFromArray(moving_seg_image)
        
        selx = sitk.ElastixImageFilter()
        selx.SetFixedImage(fixed_image) #sitk.ReadImage("fixedImage.nii")
        selx.SetMovingImage(moving_image)
        selx.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
        #overlapFilter = sitk.LabelOverlapMeasuresImageFilter()
        selx.Execute()
        moving_image=selx.GetResultImage()
        tp = selx.GetTransformParameterMap()
        tp = tp[0]
        tp["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    
        resultLabel = sitk.Transformix(moving_seg_image, tp)
    
        moving_image = sitk.GetArrayFromImage(moving_image)
        resultLabel = sitk.GetArrayFromImage(resultLabel)
        return moving_image,resultLabel
