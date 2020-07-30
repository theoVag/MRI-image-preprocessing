#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:34:43 2019

@author: Theodoros- Panagiotis Vagenas
"""
# Script for the serial execution of all preprocessing steps

import os, sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import sys,os

from FullDataPreparation import FullDataPreparation
from ImgProcess import ImgProcess
from DataSplit import DataSplit
import normImages
import img_utils
sys.path.append('/home/tpvagenas/simple/SimpleElastix/build/SimpleITK-build/Wrapping/Python/')


def clean_baseline(path):
    print("Cleaning baseline")
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".npz"):
                 #print(os.path.join(root, file))
                 os.remove(os.path.join(root, file))
    print("Baseline cleaned")


def text_open(path):
    
    ftemp = open(path,'r')
    x_files = ftemp.readlines()
    x_files = [baseline_path+'/'+x for x in x_files]
    x_files=[x.replace('\n','') for x in x_files]
    return x_files

if __name__ == "__main__":
    
    
    #Clean baseline
    baseline_path = '/home/tpvagenas/Dataset/Baseline_split1'
    #atlas_path = baseline_path+ "/KL0/9394136/9394136" #split2
    ##atlas_path = baseline_path+ "/KL0/9189553/9189553" #SPLIT1
    atlas_path = baseline_path + '/KL0/9035647/9035647'
    new_vol_size = [125,192,224]

    clean_baseline(baseline_path)
    
    # Create initial volume files (resampled)
    fdp = FullDataPreparation(path = baseline_path,new_size=[224,192],num_slices=160,resample=False,isBinary=True) # add parameters isBinary=True
    
    fdp.create_volume_files()
    fdp.create_segm_files()
    
    #img_utils.apply_flip_image(baseline_path,ns=[160,96,112])
    
    img_utils.apply_reduce_slices(baseline_path,left_size=30,right_size=50)
    
    img_utils.apply_img_resample_Base(baseline_path,ns=new_vol_size)
    
    # Apply median filter for denoising salt and pepper
    img_utils.apply_median_filter(baseline_path,1)
    
    # Denoise Image
    img_utils.apply_img_denoise_Base(baseline_path)
    
    # Histogram Matching
    img_utils.apply_histogram_matching_Base(atlas_path,baseline_path,ns = new_vol_size)
    
    
    # Contrast stretching
    img_utils.apply_constrast_enhancement_Base(baseline_path)
    
    # Normalize Image h meta to registration
    img_utils.normIm_Base(baseline_path,ns=new_vol_size)
    

    # Second processing - Affine Alignment - simpleelastix
    
    # run with different atlas its time
    
    print(baseline_path)
    #imgp = ImgProcess(baseline_path = baseline_path,atlas_path = None)
    #imgp.affine_s2s()
    imgp = ImgProcess(baseline_path = baseline_path,atlas_path = atlas_path,new_size=[224,192],num_slices=80)
    imgp.apply_affine()   
    imgp.plot_all_images(baseline_path)
    
    # Normalize Image h meta to registration
    normImages.normIm_Base(baseline_path,ns=new_vol_size)
    
    # Train/test split
    path_data = "/home/tpvagenas/main_preprocess/data_split1_ok_80"
        
    ds = DataSplit(path = baseline_path, path_data = path_data)
    #ds.dt_init_split()
    ds.apply_split()
    
    # Uncomment for filename list usage
    ##ds = DataSplit(path = baseline_path, path_data = path_data)
    ##xtrain = text_open('/home/tpvagenas/voxelmorph-master/src/train_files.txt')
    ##xval = text_open('/home/tpvagenas/voxelmorph-master/src/validate_files.txt')
    ##xtest = text_open('/home/tpvagenas/voxelmorph-master/src/test_files.txt')
    ##ds.init_from_text(xtrain=xtrain,xval=xval,xtest=xtest)
    ##ds.apply_split()


#export PYTHONPATH=$PYTHONPATH:/home/tpvagenas/simple/SimpleElastix/build/SimpleITK-build/Wrapping/Python