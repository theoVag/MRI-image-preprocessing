
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:51:14 2019

@author: Theodoros- Panagiotis Vagenas
"""

# Normalization functions and wrappers

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import SimpleITK as sitk

def normImages0_1(img,ns=[160,192,224]):
    
    img=sitk.GetImageFromArray(img)
    
    rescalFilt=sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(1)
    rescalFilt.SetOutputMinimum(0)
     
    img=rescalFilt.Execute(img)
    img=sitk.GetArrayFromImage(img).reshape(ns)
    
    return img
    

def normIm_data(path_data,ns=[160,192,224]):
    path = path_data
    train_files=glob(path+"/train/vols/*")
    validate_files=glob(path+"/validate/vols/*")
    test_files=glob(path+"/test/vols/*")
    
    print("Normalizing train files")
    apply_norm(train_files,ns)
    print("Normalizing validate files")
    apply_norm(validate_files,ns)
    print("Normalizing test files")
    apply_norm(test_files,ns)
    
    #path = "/home/tpvagenas/data"
    #path = "/home/tpvagenas/main_preprocess/data_clean"

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
                 

def apply_norm(list_files,ns):
    for i in range(len(list_files)):
        volume_image = np.load(list_files[i])['vol_data']
        #print(list_files[i])
        #print(volume_image.shape)
        #volume_image=sitk.GetImageFromArray(volume_image)
        volume_image=normImages0_1(volume_image,ns)
        #volume_image=sitk.GetArrayFromImage(volume_image).reshape((160,192,224))
        np.savez(list_files[i], vol_data=volume_image)
        
    
