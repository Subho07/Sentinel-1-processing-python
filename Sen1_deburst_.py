###############################################################################
#  Sen1_deburst.py
#
#  Project:
#  Author:   Subhadip Dey, 
#  Email:    sdey2307@gmail.com
#  Department: Agricultural and Food Engineering
#  Institute: Indian Institute of Technology Kharagpur
#  State: West Bengal
#  Country: India
#  Zipcode: 721302
#  Created:  April 2024
#
###############################################################################
#  Copyright (c) 2024, Subhadip Dey
#
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
###############################################################################

'''
This code deburst a selected swath of Sentinel-1 SLC data 
'''
import os
import numpy as np
from skimage import io
import struct
import matplotlib.pyplot as plt
import glob
import xml.etree.ElementTree as ET
import datetime
import time

def check(sentence, words):
    res = [all([k in s for k in words]) for s in sentence]
    return [sentence[i] for i in range(0, len(res)) if res[i]]

def change_2_sec(time):
    time_split = time.split(':')
    time_h = float(time_split[0])
    time_m = float(time_split[1])
    time_s = float(time_split[2])
    
    time_sec = (time_h * 60 + time_m) * 60 + time_s
    return time_sec
#%%
sen_folder = input('Provide the path of Sentinel-1 .safe file: ')
swath_sel = input('Select any one of the swaths (iw1, iw2, iw3): ').lower()
pol_sel = input('Select any one polarization (vv, vh): ').lower()

xml_folder = sen_folder + '\\annotation'
os.chdir(xml_folder)
xml_f_list = []
for file in glob.glob("*.xml"):
    xml_f_list.append(file)

keywords = [swath_sel, pol_sel]
xml_file = check(xml_f_list, keywords)[0]

xml = sen_folder + '\\annotation\\' + xml_file
tree = ET.parse(xml)
root = tree.getroot()

byteoff_burst = []

for bursts in root.iter('byteOffset'):
    byteoff_burst.append(bursts.text)
    
for linepb in root.iter('linesPerBurst'):
    lines = int(linepb.text)
    
for samplepb in root.iter('samplesPerBurst'):
    samples = int(samplepb.text)
#%%
for futc in root.iter('productFirstLineUtcTime'):
    productFirstLineUtcTime = futc.text
    productFirstLineUtcTime = productFirstLineUtcTime.split('T')[1]
    x1 = time.strptime(productFirstLineUtcTime.split('.')[0],'%H:%M:%S')
    productFirstLineUtcTime_sec = datetime.timedelta(hours=x1.tm_hour,minutes=x1.tm_min,seconds=x1.tm_sec).total_seconds()
    
for lutc in root.iter('productLastLineUtcTime'):
    productLastLineUtcTime = lutc.text
    productLastLineUtcTime = productLastLineUtcTime.split('T')[1]
    x2 = time.strptime(productLastLineUtcTime.split('.')[0],'%H:%M:%S')
    productLastLineUtcTime_sec = datetime.timedelta(hours=x2.tm_hour,minutes=x2.tm_min,seconds=x2.tm_sec).total_seconds()
    

for aztv in root.iter('azimuthTimeInterval'):
    line_time_interval = float(aztv.text)
 #%%  
firstvalidSamples = []

for fvs in root.iter('firstValidSample'):
    firstvalidSamples.append(fvs.text)
    
lastValidSample = []

for lvs in root.iter('lastValidSample'):
    lastValidSample.append(lvs.text)
    
firstvalidSamples_ = [np.fromstring(firstvalidSamples[ii], dtype=int, sep=' ') for ii in range(len(firstvalidSamples))]
firstvalidSamples_ = np.array(firstvalidSamples_)

lastValidSample_ = [np.fromstring(lastValidSample[ii], dtype=int, sep=' ') for ii in range(len(lastValidSample))]
lastValidSample_ = np.array(lastValidSample_)
#%%
azimuthTime_burst = []

for azbtime1 in root.iter('burst'): #azimuthTime
    for azbtime in azbtime1.iter('azimuthTime'):
        azimuthTime_burst.append(azbtime.text)

azimuthTime_burst = [azimuthTime_burst[ii].split('T')[1] for ii in range(len(azimuthTime_burst))]
azimuthTime_burst_ = [change_2_sec(azimuthTime_burst[ii]) for ii in range(len(azimuthTime_burst))]
#%%
tiff_folder = sen_folder + '\\measurement'
os.chdir(tiff_folder)

tiff_f_list = []
for file in glob.glob("*.tiff"):
    tiff_f_list.append(file)
    
tiff_file = check(tiff_f_list, keywords)[0]
    
sen1_IW1 = io.imread(sen_folder + '\\measurement\\' + tiff_file)
sen1_IW1_r = sen1_IW1.real
sen1_IW1_i = sen1_IW1.imag
#%%
num_output_lines, num_samples_per_line = sen1_IW1_r.shape
#%%
firstvalidSamples_flat = firstvalidSamples_.flatten()

indx_del = np.argwhere(firstvalidSamples_flat == -1)
data_r = np.delete(sen1_IW1_r, indx_del, axis = 0)
data_i = np.delete(sen1_IW1_i, indx_del, axis = 0)
firstvalidSamples_flat_clean = np.delete(firstvalidSamples_flat, indx_del, axis = 0)
#%%
az_time_diff = np.diff(azimuthTime_burst_, axis = 0)
az_samples = (az_time_diff/line_time_interval).astype(np.int16)
#%%
firstvalidSamples_ = firstvalidSamples_.astype(np.float16)
firstvalidSamples_[firstvalidSamples_ == -1] = np.NaN
count_val_samp = np.count_nonzero(~np.isnan(firstvalidSamples_), axis = 1)
#%%
data_r_deb = []
data_i_deb = []

diff = 0
start_index = 0
for ii in range(len(az_samples)):
    num_s = az_samples[ii]
    num_s = num_s - diff
    sample_burst = count_val_samp[ii]
    sample_burst = sample_burst - diff
    stop_index = sample_burst + start_index
    print(start_index, stop_index)
    data_r_deb.append(data_r[start_index: stop_index, :])
    data_i_deb.append(data_i[start_index: stop_index, :])
    diff = sample_burst - num_s
    start_index = stop_index + diff

data_r_deb.append(data_r[start_index: , :])
data_i_deb.append(data_i[start_index: , :])
#%%
data_r_deb_0 = data_r_deb[0]
data_i_deb_0 = data_r_deb[0]

for ii in range(1, len(data_r_deb), 1):
    data_r_deb_0 = np.vstack([data_r_deb_0, data_r_deb[ii]])
    data_i_deb_0 = np.vstack([data_i_deb_0, data_i_deb[ii]])
#%%   
fvs_num = int(np.nanmin(firstvalidSamples_.flatten()))
lvs_num = int(np.nanmax(lastValidSample_.flatten()))
#%%
inten = data_r_deb_0[:, fvs_num:lvs_num]**2 + data_i_deb_0[:, fvs_num:lvs_num]**2
plt.figure()
plt.imshow(inten, vmin = 30, vmax = 110)
#%%
save_folder = sen_folder + '\\python_generated'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
io.imsave(save_folder + '\\i_' + swath_sel.upper() + '_Pol_' + pol_sel.upper() + '_debursted.tiff', data_r_deb_0[:, fvs_num:lvs_num])
io.imsave(save_folder + '\\q_' + swath_sel.upper() + '_Pol_' + pol_sel.upper() + '_debursted.tiff', data_r_deb_0[:, fvs_num:lvs_num])
io.imsave(save_folder + '\\intensity_' + swath_sel.upper() + '_Pol_' + pol_sel.upper() + '_debursted.tiff', inten)
