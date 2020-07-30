#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:10:14 2020

@author: Theodoros- Panagiotis Vagenas
Script to apply all preprocessing steps to a 3d volume
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import img_utils
import normImages
import plots_lib

MAIN_PATH = os.getenv("HOME") + "/"

# function that applies all needed preprocess steps to an npz file
def PreprocessSingle(parameters):
    
    #Load every file
    volume = np.load(parameters['file_path'])[parameters['array_name']]
    seg_volume = np.load(parameters['seg_path'])[parameters['array_name']]
    
    atlas_vol = np.load(parameters['atlas_path'])[parameters['array_name']]
    
    # Cut slices from left and right side of volume
    if parameters['left_size']!=0 or parameters['right_size']!=0:
        pass
        volume = img_utils.reduce_slices(volume=volume,left_size=parameters['left_size'],right_size=parameters['right_size'])
        seg_volume = img_utils.reduce_slices(volume=seg_volume,left_size=parameters['left_size'],right_size=parameters['right_size'])
        atlas_vol = img_utils.reduce_slices(volume=atlas_vol,left_size=parameters['left_size'],right_size=parameters['right_size'])
    
    # Resize 3d volume with new width and height per slice
    if parameters['new_size']!=None:
        volume_t, seg_volume_t,atlas_vol_t = volume, seg_volume, atlas_vol
        [volume,seg_volume] = img_utils.resize_sitk(volume_image_np=volume,seg_volume_image_np=seg_volume,ns = parameters['new_size'])
        [atlas_vol,_] = img_utils.resize_sitk(volume_image_np=atlas_vol,seg_volume_image_np=seg_volume,ns = parameters['new_size'])
        
        plots_lib.plot_vols_comp(volume_t,volume,40,'resize')
    
    # Apply median filter and plot results
    volume_t, seg_volume_t = volume, seg_volume
    volume = img_utils.median_filter(inputImage = volume,rad = parameters['mf_radius'])
    atlas_vol = img_utils.median_filter(inputImage = atlas_vol,rad = parameters['mf_radius'])
    plots_lib.plot_vols_comp(volume_t,volume,40,'median')
    
    # Apply image denoising filter (Blockwise Non-Local Means) and plot results
    volume_t, seg_volume_t = volume, seg_volume
    volume = img_utils.img_denoise(data = volume, patch_rad = parameters['den'][0], block_rad = parameters['den'][1],n_sigma = parameters['den'][2])   
    atlas_vol = img_utils.img_denoise(data = atlas_vol, patch_rad = parameters['den'][0], block_rad = parameters['den'][1],n_sigma = parameters['den'][2])   
   
    plots_lib.plot_vols_comp(volume_t,volume,40,'Denoising')
    
    # Apply histogram matching
    volume_t, seg_volume_t = volume, seg_volume
    volume = img_utils.histogram_matching(image = volume,reference = atlas_vol,ns=[125,192,224]) # edw denn thelei to reference size allaghhhhhhhhhh
    plots_lib.plot_vols_comp(volume_t,volume,40,'Histogram matching')
    plots_lib.plot_comp_ref(volume_t,volume,atlas_vol,40,'Histogram matching (Histograms)')
    volume_t, seg_volume_t = volume, seg_volume
    
    # Apply constrast enhancement
    volume = img_utils.constrast_enhancement(image = volume)
    atlas_vol = img_utils.constrast_enhancement(image = atlas_vol)
    plots_lib.plot_vols_comp(volume_t,volume,40,'Constrast Enhancement')
    plots_lib.plot_hist_comp(volume_t,volume,40,'Constrast Enhancement (Histograms)')
    
    # Normalize values to [0,1]
    atlas_vol = normImages.normImages0_1(img = atlas_vol,ns = atlas_vol.shape)
    
    volume_t, seg_volume_t = volume, seg_volume
    #[volume,seg_volume] = img_utils.affine_align(moving_image = volume,moving_seg_image = seg_volume,fixed_image = atlas_vol)
    
    volume_t, seg_volume_t = volume, seg_volume
    volume = normImages.normImages0_1(img = volume,ns = volume.shape)
    plots_lib.plot_vols_comp(volume_t,volume,40,'Normalize to [0,1]')
    
    return volume, seg_volume

if __name__ == "__main__":

    
    MAIN_PATH = '/home/tpv/preprocess_clean/ants_testing/'
    parameters = dict(
        file_path = MAIN_PATH +'data_split1_ok/train/vols/9016918_test_vol.npz', 
        seg_path = MAIN_PATH +'data_split1_ok/train/asegs/9016918.segmentation_masks_test_seg.npz',
        atlas_path = MAIN_PATH +'data_split1_ok/train/vols/9035647_test_vol.npz' ,
        array_name = 'vol_data',
        left_size = 10,
        right_size = 25,
        new_size = [125,192,224],
        mf_radius = 1,
        den = [3,4,4]   
        
    )
    PreprocessSingle(parameters)
    
    
    #PreprocessSingle(**vars(args))