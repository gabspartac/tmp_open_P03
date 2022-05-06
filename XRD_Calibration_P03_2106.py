#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 00:08:30 2021

@author: Gabriel
"""
#%% Libraries


import os
import re
from pathlib import Path

import numpy as np

from fabio.edfimage import edfimage
import h5py
import hdf5plugin # For LZ4 compression reading/writing

from matplotlib import pyplot as plt

# 
from PIL import Image




#%% Path

path_proj = Path("") # current dir here

path_raw = Path(path_proj, "raw")
path_raw_edf = Path(path_proj, "raw_edf")
path_integ = Path(path_proj, "integrated")
path_res = Path(path_proj, "results")

path_ut = Path(path_proj, "utilities")


#%% Which sample to open ?
# -------------------------------------

numScan_min = 1
numScan_max = 1

numScans = range(numScan_min, numScan_max+1)

sampleName = "lab6_lastreference"

# Setup
    
E = 16.600 # keV


# -------------------------------------
# -------------------------------------



#%% Fonction


def openImage(sampleName, numScan):
    # Open the hdf5 file
    data_f = h5py.File(Path(path_raw, f'{sampleName}_{numScan:05d}', 'eiger9m', f'{sampleName}_{numScan:05d}_data_000001.h5'), 'r')
    
    # gather up the image array and decompress it (LZ4 format - done by h5py when hdf5plugin is loaded) :
    data = data_f['/entry/data/data']
    Scan = data[:]
    
    # For the metadata we need to read the .fio files (I actually don't need them...)
    #metadata = Path(path_raw, 'fiofiles', f'{sampleName}_{numScan:05d}.fio')
    
    return (Scan)



def applyMask(Scan, mask):
    
    Scan_masked = np.zeros(Scan.shape)
    
    # We ittarate on all the Scan images
    for ii in range(0, Scan.shape[0]-1):
    
        # We apply the mask
        Scan_masked[ii,:,:] = np.ma.masked_where(np.logical_not(mask), Scan[ii,:,:])

    
    return (Scan_masked)


def mkDirCheck(dirName):

    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")


numScan = 1 #dummmy
def saveImage_edf(save_image, sampleName = sampleName, numScan = numScan):
    
    if len(save_image.shape) == 2:
        numImages = 1
        
        path_dirr = Path(path_raw_edf, f'{sampleName}_{numScan:05d}')
        fileName = f'{sampleName}_{numScan:05d}_data_{numImages:05d}.edf'
        SaveFile_path = Path(path_dirr, fileName)
        
        mkDirCheck(path_dirr)
        edfimage(save_image).write(SaveFile_path)
        print(SaveFile_path)
    
    if len(save_image.shape) > 2:
        numImages = save_image.shape[0]
        print(f'{numImages} images to save...')
        
        path_dirr = Path(path_raw_edf, f'{sampleName}_{numScan:05d}')
        mkDirCheck(path_dirr)
        
        for ii in range(0, numImages):
            numImage = ii+1
            
            fileName = f'{sampleName}_{numScan:05d}_data_{numImage:05d}.edf'
            SaveFile_path = Path(path_dirr, fileName)
            
            edfimage(save_image[ii,:,:]).write(SaveFile_path)
            print(SaveFile_path)
        
        
def saveOneImage_edf(save_image, sampleName = sampleName):
    
    if len(save_image.shape) == 2:        
        path_dirr = Path(path_raw_edf, f'{sampleName}')
        fileName = f'{sampleName}.edf'
        SaveFile_path = Path(path_dirr, fileName)
        
        mkDirCheck(path_dirr)
        edfimage(save_image).write(SaveFile_path)
        print(SaveFile_path)

def saveOneImage_tiff(save_image, sampleName = sampleName):
    
    if len(save_image.shape) == 2:        
        path_dirr = Path(path_raw_edf, f'{sampleName}')
        fileName = f'{sampleName}.tiff'
        SaveFile_path = Path(path_dirr, fileName)
        
        mkDirCheck(path_dirr)
        im = Image.fromarray(save_image)
        im.save(SaveFile_path)
        print(SaveFile_path)

        
        
_to_esc = re.compile(r'\s|[]()[]')
def _esc_char(match):
    return '\\' + match.group(0)
 
def my_escape(name):
    return _to_esc.sub(_esc_char, name)
        
#%% Opening the mask file

mask_filename = 'Eiger_X_9M_mask.tif' # just the dead areas

mask = Image.open(Path(path_ut, 'mask_files', mask_filename))
mask = np.array(mask)
mask = mask > 0


#%% Open a Map and plot an image

Scan = openImage(sampleName, numScans[0]) # preliminary open the first image
Map = np.zeros([len(numScans), Scan.shape[0], Scan.shape[1], Scan.shape[2]])

# Construction of the 2D map
for ii in range(0, len(numScans)):
    numScan = numScans[ii]
    
    Scan = openImage(sampleName, numScan)
    Map[ii,:,:,:] = Scan

# We apply the mask of the Eiger
#maskedScan = applyMask(Scan, mask)

#-----------
# Open the image
numScan_map = 0
numImage = 0

lims = [np.min(Map[numScan_map,numImage, :,:]), 0.1*np.max(Map[numScan_map,numImage, :,:])]
lims = [0, 5]

# Log
#plt.imshow(np.log10(Scan[numImage, :,:]), cmap='jet')
#plt.clim(np.log(lims[0]+1), lims[1])
#plt.show()

# Lin
plt.imshow(Map[numScan_map,numImage, :,:], cmap='jet')
plt.clim(lims[0], lims[1])
plt.show()

#%% We can improove a bit the mask with the dead pixels

mask = mask | (Scan[0,:,:] > 4294967295 - 1)



#%% Averaging the scan of the LaB6

# On one Scan
#Averaged_Image = np.mean(Scan, 0)

# Over the whole 2D map
Averaged_Scan = np.zeros([Map.shape[0], Map.shape[2], Map.shape[3]])
for ii in range(0, len(numScans)):
    Averaged_Scan[ii,:,:] = np.mean(Map[ii,:,:,:], 0)

Averaged_Image = np.mean(Averaged_Scan, 0)

#-----------
# Open the averaged image
lims = [0, 30]

# Log
#plt.imshow(np.log10(Scan[numImage, :,:]), cmap='jet')
#plt.clim(np.log(lims[0]+1), lims[1])
#plt.show()

# Lin
plt.imshow(Averaged_Image, cmap='jet')
plt.clim(lims[0], lims[1])
plt.show()

#%% Save images in .edf format for the calibration with pyFAI


# Mask saving
# SaveFile_path = Path(path_ut, 'mask_files', 'Eiger_X_9M_mask.edf')
# edfimage(1*mask).write(SaveFile_path)
# print(SaveFile_path)

# Saving averaged image of LaB6
#saveImage_edf(Averaged_Image)
saveOneImage_edf(Averaged_Image)

#saveOneImage_tiff(Averaged_Image)



#%% Calibration


detector = 'eiger9m'

# SAXS
        
calib_IMG =  Path(path_raw_edf, f'{sampleName}', f'{sampleName}.edf')

calib_IMG = str(calib_IMG)
calib_IMG = my_escape(calib_IMG)

mask_IMG = Path(path_ut, 'mask_files', 'mask_no_indenter.edf')
mask_IMG = str(mask_IMG)
mask_IMG = my_escape(mask_IMG)


cmd = f"pyFAI-calib2 -e {E} --calibrant LaB6 -D {detector} -m {mask_IMG}  {calib_IMG}"


os.system(cmd)







#%%






