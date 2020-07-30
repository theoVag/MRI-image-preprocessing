#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:25:50 2019

@author: Theodoros- Panagiotis Vagenas
"""

import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import SimpleITK as sitk

# Class for affine aligning all images in Baseline folder
class ImgProcess:
    """Description here"""
    
    def __init__(self
                 , baseline_path
                 , atlas_path
                 , new_size=[224,192]
                 , num_slices=160
                 , resample = True
                 
                  ):
        self.baseline_path = baseline_path
        self.atlas_path=atlas_path
        self. data_dirs = []
        #self.new_size=new_size
        #self.num_slices=num_slices
        if atlas_path != None:
            self.atlas_vol = np.load(atlas_path +'_test_vol.npz')['vol_data']
            self.atlas_seg = np.load(atlas_path+'.segmentation_masks_test_seg.npz')['vol_data']
    
    # Affign align both image and segmentation map
    def affine_align(self,moving_image,moving_seg_image):
        
        moving_image=sitk.GetImageFromArray(moving_image)
        fixed_image=sitk.GetImageFromArray(self.atlas_vol)
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
    
    # Wrapper for affine function to Baseline folder
    def apply_affine(self):
        print ("Affine alligning...")
        baseline_path= self.baseline_path
        kl_dirs=glob(baseline_path+"/*/")
        if (baseline_path + '/doc/' in kl_dirs):
            kl_dirs.remove(baseline_path + '/doc/')
            
        data_dirs = glob(baseline_path+"/*/*/")
        
        self. data_dirs= data_dirs
        for directory in data_dirs:
            filename = directory.split("/")[-2]
            print(filename)
            volume_image = np.load(directory + filename+'_test_vol.npz')['vol_data']
            seg_image = np.load(directory + filename+'.segmentation_masks_test_seg.npz')['vol_data']
            vol_aligned, seg_aligned = self.affine_align(volume_image,seg_image)
            if directory == data_dirs[15]:
                pass #self.test_result(volume_image,seg_image,vol_aligned,seg_aligned)

            np.savez(directory + filename+'_test_vol.npz', vol_data=vol_aligned)
            np.savez(directory + filename+'.segmentation_masks_test_seg.npz', vol_data=seg_aligned)
    
    # Affine align subject-to-subject, all pairs (optional)
    def affine_s2s(self):
        baseline_path = self.baseline_path
        data_dirs = glob(baseline_path+"*.npz")
        print(data_dirs)
        for at_path in data_dirs:
            print(at_path)
            filename = at_path.split("/")[-2]
            self.atlas_vol = np.load(at_path)['vol_data']
            t= at_path.replace('_test_vol.npz','.segmentation_masks_test_seg.npz')
            print(t)
            #self.atlas_seg = np.load(t.replace('vols','asegs'))['vol_data']
            for directory in data_dirs:
                filename = directory.split("/")[-2]
                vol_name = directory
                seg_name = vol_name.replace('_test_vol.npz','.segmentation_masks_test_seg.npz')
                seg_name = seg_name.replace('vols','asegs')
                volume_image = np.load(vol_name)['vol_data']
                seg_image = np.load(seg_name)['vol_data']
                
                vol_aligned, seg_aligned = self.affine_align(volume_image,seg_image)
                if directory == data_dirs[15]:
                    pass #self.test_result(volume_image,seg_image,vol_aligned,seg_aligned)
                #vol_aligned = volume_image
                #seg_aligned = seg_image
                #break
                ##np.savez(vol_name, vol_data=vol_aligned)
                ##np.savez(seg_name, vol_data=seg_aligned)
    
    # Save to image atlas and volume
    def test_result(self,volume_image,seg_image,vol_aligned,seg_aligned):
        slice_coord = [60,30,60]
        atlas_vol = self.atlas_vol
        fig, axes = plt.subplots(2, 3, figsize = (20, 10))
        fig.suptitle('Original volume, segmentation - mean,max')
        axes[0,0].imshow(volume_image[slice_coord[0],:,:])
        axes[0,0].title.set_text('Initial Volume')
        axes[0,1].imshow(atlas_vol[slice_coord[0],:,:])
        axes[0,1].title.set_text('atlas_vol')
        axes[0,2].imshow((atlas_vol-volume_image)[slice_coord[0],:,:])
        axes[0,2].title.set_text('diff(atlas-initial)')
        axes[1,0].imshow(vol_aligned[slice_coord[0],:,:])
        axes[1,0].title.set_text('Volume aligned')
        axes[1,1].imshow(seg_aligned[slice_coord[0],:,:])
        axes[1,1].title.set_text('Segm aligned')
        axes[1,2].imshow((atlas_vol-vol_aligned)[slice_coord[0],:,:])
        axes[1,2].title.set_text('diff(atlas-aligned)')          
        fig.savefig('test.jpg')
    
    # Plot all images and segmentations for testing    
    def plot_all_images(self,baseline_path):
        from natsort import index_natsorted

        #baseline_path= self.baseline_path
        kl_dirs=glob(baseline_path+"/*/")
        if (baseline_path + '/doc/' in kl_dirs):
            kl_dirs.remove(baseline_path + '/doc/')
            
        data_dirs = glob(baseline_path+"/*/*/")
        slice_coord = [60,30,60]
        cols=4
        rows = int(np.ceil(len(data_dirs)/cols))
        fig, axes = plt.subplots(rows, cols, figsize = (20, 60), gridspec_kw = {'wspace':0.1, 'hspace':0.3})
        fig.suptitle('Original volumes')
        #plt.subplots_adjust(wspace=0.1, hspace=0.1)
        fig.subplots_adjust(hspace=0.0, wspace=0.0)
        
        fig2, axes2 = plt.subplots(rows, cols, figsize = (20,60), gridspec_kw = {'wspace':0.1, 'hspace':0.3})
        fig2.suptitle('Original segmentations')
        fig2.subplots_adjust(hspace=0.0, wspace=0.0)
        
        ind=0
        data_kl = [directory.split("/")[-3] for directory in data_dirs]
        #print(data_kl)
        data_dirs_index = index_natsorted(data_kl)
        #print(data_dirs_index)
        dt=[]
        for i in range(len(data_dirs_index)):
            dt.append(data_dirs[data_dirs_index[i]])
        
        
        data_dirs = dt
        
        for directory in data_dirs:
            iy= int(ind % cols)
            ix = int(np.floor(ind /cols))
            kl_ind = directory.split("/")[-3]
            #print("\n ix=%f    iy=%f" % (ix,iy))
            filename = directory.split("/")[-2]
            #print(directory.split("/")[-3])
            volume_image = np.load(directory + filename+'_test_vol.npz')['vol_data']
            seg_image = np.load(directory + filename+'.segmentation_masks_test_seg.npz')['vol_data']
            axes[ix,iy].imshow(volume_image[slice_coord[0],:,:])
            axes[ix,iy].set_title('vol %s - %s' % (filename,kl_ind))
            
            axes2[ix,iy].imshow(seg_image[slice_coord[0],:,:])
            axes2[ix,iy].set_title('vol %s - %s' % (filename,kl_ind))
            ind=ind+1
        
        #fig.subplots_adjust(wspace=0.1, hspace=0.3)
        fig.savefig('/home/tpv/test_vol_all.jpg')
        #fig2.subplots_adjust(wspace=0.1, hspace=0.3)
        fig2.savefig('/home/tpv/test_seg_all.jpg')
