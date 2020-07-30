#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:34:49 2020

@author: Theodoros- Panagiotis Vagenas
"""
# Plots for image preprocess analysis and verification

import matplotlib.pyplot as plt
import numpy as np


def plot_vols_comp(vol_in,vol_out,slice_num,filt_name):
    
    fig, axes = plt.subplots(1, 2, figsize = (20, 10))
    fig.suptitle('Results for filter: %s' % filt_name)
    
    z00_plot=axes[0].imshow(vol_in[slice_num,:,:],cmap='gray')
    axes[0].title.set_text('Initial Volume')
    plt.colorbar(z00_plot,ax=axes[0])

    
    z01_plot=axes[1].imshow(vol_out[slice_num,:,:],cmap='gray')

    axes[1].title.set_text('Output Volume')
    plt.colorbar(z01_plot,ax=axes[1])
    
    fig.savefig(filt_name+'.jpg')
    
    
def plot_hist_comp(vol_in,vol_out,slice_num,filt_name):
    
    fig, axes = plt.subplots(1, 2, figsize = (20, 10))
    fig.suptitle('Results for filter: %s' % filt_name)
    
    z00_plot=axes[0].hist(vol_in[slice_num,:,:])
    axes[0].title.set_text('Initial Histogram')

    
    z01_plot=axes[1].hist(vol_out[slice_num,:,:])

    axes[1].title.set_text('Output Histogram')
    
    fig.savefig(filt_name+'.jpg')
    
def plot_hist_comp_ref(vol_in,vol_out,ref,slice_num,filt_name):
    
    fig, axes = plt.subplots(1, 3, figsize = (20, 10))
    fig.suptitle('Results for filter: %s' % filt_name)
    
    z00_plot=axes[0].hist(vol_in[slice_num,:,:])
    axes[0].title.set_text('Initial Histogram')

    
    z01_plot=axes[1].hist(ref[slice_num,:,:])

    axes[1].title.set_text('Reference Histogram')
    
    z01_plot=axes[2].hist(vol_out[slice_num,:,:])

    axes[2].title.set_text('Output Histogram')
    
    fig.savefig(filt_name+'.jpg')
    
    
def plot_comp_ref(vol_in,vol_out,ref,slice_num,filt_name):
    
    
    fig, axes = plt.subplots(2, 3, figsize = (20, 10))
    fig.suptitle('Results for filter: %s' % filt_name)
    
    z00_plot=axes[0,0].imshow(vol_in[slice_num,:,:],cmap='gray')
    axes[0,0].title.set_text('Initia Volume')
    #plt.colorbar(z00_plot,ax=axes[0,0])
    
    z01_plot=axes[0,1].imshow(ref[slice_num,:,:],cmap='gray') #cmap= YlGnBu #bwr
    axes[0,1].title.set_text('Reference Volume')
    
    z02_plot=axes[0,2].imshow(vol_out[slice_num,:,:],cmap='gray')
    #plt.colorbar(z02_plot,ax=axes[0,2])
    axes[0,2].title.set_text('Output Segmentation')

    
    z10_plot=axes[1,0].hist(vol_in[slice_num,:,:])
    axes[1,0].title.set_text('Initial Histogram')
    #plt.colorbar(z10_plot,ax=axes[1,0])
    
    z11_plot=axes[1,1].hist(ref[slice_num,:,:])
    axes[1,1].title.set_text('Reference Histogram')
    #plt.colorbar(z11_plot,ax=axes[1,1])
    
    z12_plot=axes[1,2].hist(vol_out[slice_num,:,:])
    axes[1,2].title.set_text('Output Histogram')

    
    fig.savefig(filt_name+'.jpg')