#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:06:16 2019

@author: Theodoros - Panagiotis Vagenas
"""

import os
import pydicom
#import dicom_numpy
import numpy as np
#import matplotlib.pyplot as plt
from glob import glob
import SimpleITK as sitk
from PIL import Image as image
import natsort 
import img_utils

class FullDataPreparation:
    """Class to read dicom and mhd files, prepare them and save them as npz files. This type is required from the framework"""
    
    # Initialize lists for volumes and segmentation from Baseline folder
    def __init__(self
                 , path
                 , new_size=[224,192]
                 , num_slices=160
                 , resample = True
                 , isBinary = False
                  ):
        self.path = path
        self.resample=resample
        self.new_size=new_size
        self.num_slices=num_slices
        self.isBinary=isBinary
        #if var1 == None:
        #    self.var1 = 0
    
        # assert .... , 'Error message'
        self.directories=[]
        self.dir_list=[]
        self.seg_list=[]
        self.vol_list=[]
        
        kl_dirs=glob(path+"/*/")
        if (path + '/doc/' in kl_dirs):
            kl_dirs.remove(path + '/doc/')
        
        self.kl_dirs=kl_dirs
        # Extract mhd files and their directory paths
        segm_files_list = []
        segm_files_dir = []
        for kl_path in kl_dirs:

            for directory in glob(kl_path+"/*/"):
                for mhd_file in glob(directory +"*.mhd" ):

                    segm_files_list.append(mhd_file)
                    segm_files_dir.append(directory)
       
        self.segm_files_list = segm_files_list
        self.segm_files_dir = segm_files_dir

    # Extract array from dicom files
    def extract_voxel_data(self,list_of_dicom_files):
    
        datasets = [pydicom.read_file(f) for f in list_of_dicom_files]
        datasets = sorted(datasets, key=lambda x: float(x.SliceLocation)) 
        voxel_ndarray = np.zeros((len(datasets),datasets[0].Rows,datasets[0].Columns))
        
        for i in range(len(datasets)):
            voxel_ndarray[i,:,:] =  datasets[i].pixel_array
            #voxel_ndarray[i,:,:] = np.flipud(voxel_ndarray[i,:,:])
        
        return voxel_ndarray
    
    # mallon den douleuei
    def resample_image2_seg(self,numpyImage,new_size,num_slices):
        #resampled_3dimage = np.zeros((new_size[0],new_size[1],num_slices))
        resampled_3dimage = np.zeros((num_slices,new_size[1],new_size[0]))
        for i in range(0,num_slices):
            
            img = image.fromarray(numpyImage[:,:,i]*60,'L')
            img = img.resize((new_size[0],new_size[1]), image.NEAREST) #BILINEAR BICUBIC ANTIALIAS NEAREST
            #resampled_3dimage[i,:,:] = img
            resampled_3dimage[i,:,:] = np.flipud(img)
        return resampled_3dimage
    
    # mallon den douleuei
    def resample_image2(self,numpyImage,new_size,num_slices):
        #resampled_3dimage = np.zeros((new_size[0],new_size[1],num_slices))
        resampled_3dimage = np.zeros((num_slices,new_size[1],new_size[0]))
        for i in range(0,num_slices):
            img = image.fromarray(numpyImage[i,:,:]*60) #np.rot90  
            img = img.resize((new_size[0],new_size[1]), image.BILINEAR) #BILINEAR BICUBIC ANTIALIAS NEAREST
            resampled_3dimage[i,:,:] = img
    
        return resampled_3dimage
    
    # mallon den douleuei
    def resize_sitk(self,volume_image_np,seg_volume_image_np,nsize,isLabel=False):
        #new_size = [112,96,160]
        
        #num_slices = volume_image_np.shape[0]
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
        
        # resample labels
        """tr = resample.GetTransform()
        label_resample = sitk.ResampleImageFilter()
        label_resample.SetTransform(tr)"""
        
        resample.SetInterpolator = sitk.sitkNearestNeighbor
        resampled_label_image = resample.Execute(labeled_image)
        
        resampled_label_image = sitk.GetArrayFromImage(resampled_label_image)
        #print(new_image.shape)
        num_slices = new_image.shape[0]
        """if isLabel:
            for i in range(num_slices-1):
                new_image[i,:,:] = np.flipud(new_image[i,:,:])"""
        
        
        return new_image,resampled_label_image
    
    # Flip segmentation if required
    def flip_segm(self,numpyImage):
        new_size=numpyImage.shape[0:2]
        num_slices=numpyImage.shape[2]
        resampled_3dimage = np.zeros((num_slices,new_size[1],new_size[0]))
        for i in range(0,num_slices):        
            resampled_3dimage[i,:,:] = np.flipud(numpyImage[:,:,i])
    
        return resampled_3dimage
    
    # Create segmentation files from mhd files, works only for this Baseline folder structure 
    def create_segm_files(self):
        print("Creating segmentation mask files...")
        segm_files_dir=self.segm_files_dir
        segm_files_list=self.segm_files_list
        resample=self.resample
        seg_list =[]
        #print(segm_files_list)
        for d,mask in zip(segm_files_dir,segm_files_list):
            itkimage = sitk.ReadImage(mask)
            numpy_itkimage_init = sitk.GetArrayFromImage(itkimage)
            
            new_size=self.new_size
            num_slices=self.num_slices
            ns = new_size + [num_slices]

            numpy_itkimage = np.zeros([num_slices,numpy_itkimage_init.shape[0],numpy_itkimage_init.shape[1]])#mporei seira1-0
            
            """for i in range(num_slices):
                numpy_itkimage[i,:,:] =  np.flipud(numpy_itkimage_init[:,:,i])""" # prin anoixto

            if (resample==True):
                #numpy_itkimage=self.resample_image2_seg(numpy_itkimage,new_size,num_slices)
                #file1.write(str(numpy_itkimage.shape))
                ##numpy_itkimage=self.resize_sitk(numpy_itkimage,ns,True)
                #file1.write(str(ns))
                #file1.write(str(numpy_itkimage.shape))
                #file1.close()
                pass
            else:
                numpy_itkimage=self.flip_segm(numpy_itkimage_init)
                #pass
            #numpy_itkimage=change_channel_order(numpy_itkimage)
            #numpy_itkimage = img_utils.binary_mask(numpy_itkimage)
            if self.isBinary==True:
                numpy_itkimage = img_utils.binary_mask(numpy_itkimage)
            
            seg_list.append(os.path.splitext(mask)[0]+'_test_seg.npz')
            numpy_itkimage = np.array(numpy_itkimage,dtype='uint8')
            np.savez(os.path.splitext(mask)[0]+'_test_seg.npz', vol_data=numpy_itkimage)
        
        self.seg_list=seg_list.copy()
        print("End of process: segmentation mask files")
        
    # Create volume files from dicom, works only for this Baseline folder structure    
    def create_volume_files(self):
        print("Creating volume files...")
        kl_dirs=self.kl_dirs
        resample=self.resample
        new_size=self.new_size
        num_slices=self.num_slices
        dicom_paths = []
        patient_list_path=[]
        ns = new_size + [num_slices]
        for kl_path in kl_dirs:
            for directory in glob(kl_path+"/*/"):  
                patient_list_path.append(directory)
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file=='001':
                            dicom_paths.append(os.path.dirname(os.path.join(root, file)))
        vol_list=[]                   
        for p,directory in zip(dicom_paths,patient_list_path):
            file_list = glob(p + "/*")
            file_list = natsort.natsorted(file_list)
            
            full_image = self.extract_voxel_data(file_list)
            file1 = open("MyFile.txt","a") 
            
            if (resample==True):
                #full_image=self.resample_image2(full_image,new_size,num_slices)
                """file1.write(str(full_image.shape))
                full_image=self.resize_sitk(full_image,ns,False)
                file1.write(str(ns))
                file1.write(str(full_image.shape))
                file1.close()"""
            
            vol_list.append(os.path.dirname(directory)+"/"+directory.split("/")[-2]+'_test_vol.npz')
            np.savez(os.path.dirname(directory)+"/"+directory.split("/")[-2]+'_test_vol.npz', vol_data=full_image)
        
        self.vol_list=vol_list
        print("End of volume files...")
    

    