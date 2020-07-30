#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:13:03 2019

@author: Theodoros- Panagiotis Vagenas
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from shutil import copyfile,rmtree

class DataSplit:
    """Class to split initial files to train/validation/test randomly or according to file name list"""
    
    def __init__(self
                 , path
                 , path_data
                 
                  ):
        self.path = path
        self.path_data = path_data
        self.data_dirs=[]
        self.X_train=[]
        self.X_val=[]
        self.X_test=[]
        self.y_train=[]
        self.y_val=[]
        self.y_test=[]

        # Create required folders for he framework
        if os.path.exists(path_data+"/train/vols"):
            rmtree(path_data+"/train/vols")
        os.makedirs(path_data+"/train/vols")
        
        if os.path.exists(path_data+"/train/asegs"):
            rmtree(path_data+"/train/asegs")
        os.makedirs(path_data+"/train/asegs")
        if os.path.exists(path_data+"/validate/vols"):
            rmtree(path_data+"/validate/vols")
        os.makedirs(path_data+"/validate/vols")
        
        if os.path.exists(path_data+"/validate/asegs"):
            rmtree(path_data+"/validate/asegs")
        os.makedirs(path_data+"/validate/asegs")
        
        if os.path.exists(path_data+"/test/vols"):
            rmtree(path_data+"/test/vols")
        os.makedirs(path_data+"/test/vols")
        
        if os.path.exists(path_data+"/test/asegs"):
            rmtree(path_data+"/test/asegs")
        os.makedirs(path_data+"/test/asegs")
        
    # Scan Baseline and split proportional for each class and for train/val/test (stratified split)    
    def dt_init_split(self):
        path = self.path
        kl_dirs=glob(path+"/*/")
        if (path + '/doc/' in kl_dirs):
            kl_dirs.remove(path + '/doc/')    
        data_dirs = glob(path+"/*/*/")
        
        data_dirs[0].split('/')[-3]
        
        y_label = []
        for d in range(len(data_dirs)):
            y_label.append( data_dirs[d].split('/')[-3])
        
        X_train, X_mid, y_train, y_mid = train_test_split( data_dirs, y_label,stratify=y_label,test_size=0.87, random_state=1122)
            
        X_val, X_test, y_val, y_test = train_test_split( X_mid, y_mid,test_size=0.4, random_state=422)
        
        # Create list files for afterwards use
        fop = open(self.path_data+"/train_files.txt",'w+')
        for pp in X_train:
            temp = pp.split('/')
            temp =temp[len(temp)-3:len(temp)-1]
            temp = '/'.join(temp)
            fop.write("%s/\n" % temp)
        
        fop = open(self.path_data+"/validate_files.txt",'w+')
        for pp in X_val:
            temp = pp.split('/')
            temp =temp[len(temp)-3:len(temp)-1]
            temp = '/'.join(temp)
            fop.write("%s/\n" % temp)
        
        fop = open(self.path_data+"/test_files.txt",'w+')
        for pp in X_test:
            temp = pp.split('/')
            temp =temp[len(temp)-3:len(temp)-1]
            temp = '/'.join(temp)
            fop.write("%s/\n" % temp)
        
        self.data_dirs=data_dirs
        self.X_train=X_train
        self.X_val=X_val
        self.X_test=X_test
        self.y_train=y_train
        self.y_val=y_val
        self.y_test=y_test
    
    # Wrapper to copy files in the required paths from texts
    def copy_files(self,data,dir_name):
        print("Copying files")
        print(len(data))
        for sample in range(len(data)):
            file_path_vol = glob(data[sample] + "*_test_vol.npz")[0]
            file_path_seg = glob(data[sample] + "*_test_seg.npz")[0]
            copyfile(file_path_vol, self.path_data+dir_name+ "vols/" + file_path_vol.split('/')[-1])
            copyfile(file_path_seg, self.path_data+dir_name+ "asegs/" + file_path_seg.split('/')[-1])
    
    # Wrapper to copy files for train/val/test    
    def apply_split(self):
        
        # Create train
        print("Copying train files")
        self.copy_files(self.X_train,'/train/')
        print("Copying validate files")
        self.copy_files(self.X_val,'/validate/')
        print("Copying test files")
        self.copy_files(self.X_test,'/test/')
    
    # Wrapper in case that list of files are given
    def init_from_text(self,xtrain,xval,xtest):
        self.X_train=xtrain
        self.X_val=xval
        self.X_test=xtest
        
        
