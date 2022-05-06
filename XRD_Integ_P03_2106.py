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
import pyFAI

from fabio.edfimage import edfimage
import fabio
import h5py
import hdf5plugin

from matplotlib import pyplot as plt
import matplotlib


from tqdm import tqdm



#%% Path

path_proj = Path("") #♣current

path_raw = Path(path_proj, "raw")
path_raw_edf = Path(path_proj, "raw_edf")
path_integ = Path(path_proj, "integrated")
path_res = Path(path_proj, "results")

path_ut = Path(path_proj, "utilities")


#%% Which sample to open ?
# -------------------------------------

# Sample
# ----
sampleName = "lab6_lastreference"

numScan_min = 1
numScan_max = 1

Azim_sections = 1 # Number of azimuthal sections (1 = no azimutal partitioning)

pix_per_point = 0.5 # Number of pixel on the detecor per integration point (can be less than 1)
integ_radial_rng = [14, 36] # Radial range of the azimythal integration

safe_integ = True # False is faster | /!\ NEVER set True when Azim_sections > 1 --> BOG

numScans = range(numScan_min, numScan_max+1) # don't touch that
Azimuth_step = 360 / Azim_sections # don't touch that

# Setup
# ----
E = 16.600 # keV

poni_fileName = 'Geometry_carbides.poni'
mask_fileName = 'mask_with_indenter.edf'

detector = 'eiger9m'
# -------------------------------------
# -------------------------------------

print("------------------------------------------------------------------------------")
print("***********************")
print("***********************")
print("-")
print(f'Sample is: {sampleName}')
print("-")
print(f'Unsing Scan {numScan_min} to {numScan_max} for the mapping')
print("-")
if Azim_sections > 1:
    print(f'Azimutal partitionning with {Azim_sections} steps ({Azimuth_step}°/step)')
    print("-")
print("------------------------")

#%% Fonctions


def openImage(sampleName, numScan):
    # Open the hdf5 file
    data_f = h5py.File(Path(path_raw, f'{sampleName}_{numScan:05d}', 'eiger9m', f'{sampleName}_{numScan:05d}_data_000001.h5'), 'r')
    
    # gather up the image array and decompress it (LZ4 format) :
    Compressed_Scan = data_f['/entry/data/data']
    Scan = Compressed_Scan[:]
    
    # For the metadata we need to read the .fio files
    #metadata = Path(path_raw, 'fiofiles', f'{sampleName}_{numScan:05d}.fio')
    
    return (Scan)



def applyMask(Scan, mask):
    
    Scan_masked = np.zeros(Scan.shape)
    
    # We ittarate on all the Scan images
    for ii in range(0, Scan.shape[0]-1):
    
        # We apply the mask
        Scan_masked[ii,:,:] = np.ma.masked_where(np.logical_not(mask), Scan[ii,:,:])

    
    return (Scan_masked)


def mkDirCheck(dirName, muted = False):
 
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        if not(muted):
            print("Directory " , dirName ,  " Created ")
    else:    
        if not(muted):
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

        
        
_to_esc = re.compile(r'\s|[]()[]')
def _esc_char(match):
    return '\\' + match.group(0)
 
def my_escape(name):
    return _to_esc.sub(_esc_char, name)
        

def get_lambda(E):
    # Energy should be in keV !!!
    # Return the wavelenght in m
    # Plank Constant
    h = 6.62607004e-34 # m2 . kg / s
    # light celerity
    c = 299792458 # m / s
    # Conversion Joule -> eV
    eV = 6.242e+18 # eV / J
    
    # Conversion Energy -> wavelenght 
    wavelenght = ((h*c)/(E*1e3 / eV))  # m
    
    return wavelenght

#%% Opening the mask file

mask = fabio.open(Path(path_ut, 'mask_files', mask_fileName))
mask = mask.data
mask = mask > 0

print("***********************")
print(f'Using mask file: {mask_fileName}')
print("------------------------")

#%% Open the Map of images
print('Opening all the scans of the mapping...')

Scan = openImage(sampleName, numScans[0]) # preliminary open the first image
Map = np.zeros([len(numScans), Scan.shape[0], Scan.shape[1], Scan.shape[2]])
Map[0,:,:,:] = Scan

# Construction of the 2D map
for ii in tqdm(range(1, len(numScans))):
    numScan = numScans[ii]
    
    Map[ii,:,:,:] = openImage(sampleName, numScan)


#-----------
#%% Open the image
numScan_map = 0
numImage = 0

lims = [np.min(Map[numScan_map,numImage, :,:]), 0.1*np.max(Map[numScan_map,numImage, :,:])]
lims = [0, 10]

# Log
# plt.imshow(np.log10(Scan[numImage, :,:]), cmap='jet')
# plt.clim(np.log(lims[0]+1), lims[1])
# plt.show()

# Lin
plt.imshow(Map[numScan_map,numImage, :,:], cmap='jet')
plt.clim(lims[0], lims[1])
plt.show()


#%% Save images in .edf format (just in case, normally there is no need to do so)

# Saving the images of all the scans
#for ii in range(0, Map.shape[0]):
#    numScan = numScans[ii]
#    saveImage_edf(Map[ii,:,:,:], sampleName, numScan)

#%% Loading the poni file

# Reading the poni file (containing the geometry)
geometry_path = Path(path_ut, 'poni' , poni_fileName)
ai_WAXS = pyFAI.load(str(geometry_path))


print("***********************")
print(f'Using geometry: {poni_fileName}')
print("------------------------")

#%% Integrating !

# Documentation ai.integrate1d() :
# https://pyfai.readthedocs.io/en/latest/usage/cookbook/integration_with_python.html?highlight=ai.integrate1d
# https://pyfai.readthedocs.io/en/master/api/pyFAI.html
# -------

# Wavelenght
wavelenght = get_lambda(E) # Conversion E --> lambda

ai_WAXS.set_wavelength(wavelenght)


# Determination of the umber of integration points
pix_size = ai_WAXS.pixel1 # m
D = ai_WAXS.dist # m
Dtheta_pix = np.arctan(pix_size/D) * (180/np.pi) # deg

radial_lenght = integ_radial_rng[1] - integ_radial_rng[0] # deg
nb_point = np.floor(radial_lenght/(pix_per_point * Dtheta_pix))

print(f"{pix_per_point} pixel(s) per point of integration")
print(f"{nb_point} points used for the integration")


Azimuth_RNG = [0, 0] # Initialisation

# On fait la correction de flat
#IMG_SAXS = np.divide(IMG_SAXS, flat_SAXS)

print('----------------------------------------------')
print('Integration in progres...')


for ii in tqdm(range(0,Map.shape[0]), colour='red'):
    for jj in tqdm(range(0,Map.shape[1]), leave=False):
    
        IMG = Map[ii,jj,:,:]
        
        numScan = numScans[ii]
        numImage = jj+1
        
        path_file = Path(path_integ, f'{sampleName}', f'{sampleName}_Scan_{numScan:05d}')
        
        
        mkDirCheck(Path(path_integ, f'{sampleName}'), True)
        mkDirCheck(path_file, True)
        
        if Azim_sections > 1:
            
            for kk_azm in range(0, Azim_sections):
                
                filename = f'{sampleName}_Scan_{numScan:05d}_Image_{numImage:05d}_no_{kk_azm:05d}.dat'
                saving_path = Path(path_file, filename)
                
                Azimuth_RNG = [kk_azm * Azimuth_step, Azimuth_step + (kk_azm * Azimuth_step)]
                #print(Azimuth_RNG)
                
                res_WAXS = ai_WAXS.integrate1d(IMG,
                                     npt = nb_point,
                                     azimuth_range = Azimuth_RNG,
                                     filename = str(saving_path),
                                     correctSolidAngle = True,
                                     error_model = 'azimuthal', 
                                     radial_range = integ_radial_rng,
                                     mask = mask, # 1 for masked pixels, and 0 for valid pixels
                                     polarization_factor = None,
                                     unit = "2th_deg",
                                     safe = safe_integ,
                                     normalization_factor = 1)
            
            
            
        else:
            
            filename = f'{sampleName}_Scan_{numScan:05d}_Image_{numImage:05d}.dat'
            saving_path = Path(path_file, filename)
            
            res_WAXS = ai_WAXS.integrate1d(IMG,
                                 npt = nb_point,
                                 filename = str(saving_path),
                                 correctSolidAngle = True,
                                 error_model = 'azimuthal', 
                                 radial_range = integ_radial_rng,
                                 mask = mask, # 1 for masked pixels, and 0 for valid pixels
                                 polarization_factor = None,
                                 unit = "2th_deg",
                                 safe = safe_integ,
                                 normalization_factor = 1)
            
print("--------")
print('done !')
