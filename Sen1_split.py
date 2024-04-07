###############################################################################
#  Sen1_split.py
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
This code split a selected burst from a selected swath of Sentinel-1 SLC data 
'''
import os
import numpy as np
from skimage import io
import struct
import matplotlib.pyplot as plt
import glob
import xml.etree.ElementTree as ET

def check(sentence, words):
    res = [all([k in s for k in words]) for s in sentence]
    return [sentence[i] for i in range(0, len(res)) if res[i]]

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
num_bursts = np.linspace(1, len(byteoff_burst), num = len(byteoff_burst))
print('Number of bursts per swath: ', len(num_bursts))
#%%
tiff_folder = sen_folder + '\\measurement'
os.chdir(tiff_folder)

tiff_f_list = []
for file in glob.glob("*.tiff"):
    tiff_f_list.append(file)
    
tiff_file = check(tiff_f_list, keywords)[0]
    
sen1_IW1 = io.imread( sen_folder + '\\measurement\\' + tiff_file)
sen1_IW1_r = sen1_IW1.real
sen1_IW1_i = sen1_IW1.imag
sen1_IW1_r = sen1_IW1_r.flatten()
sen1_IW1_i = sen1_IW1_i.flatten()
sen1_IW1_r_bytes = sen1_IW1_r.tobytes()
sen1_IW1_i_bytes = sen1_IW1_i.tobytes()

tot_byte = len(sen1_IW1_i_bytes)

sel_burst = int(input('Enter burst number: ')) - 1

deduct_offset = int(byteoff_burst[0]) + 1
byte_start = int(byteoff_burst[sel_burst]) - deduct_offset + 1

if sel_burst == len(byteoff_burst) - 1:
    byte_stop = tot_byte
    
else:
    byte_stop = int(byteoff_burst[sel_burst + 1]) - deduct_offset

data_r = []
data_i = []

for ii in range(byte_start, byte_stop, 4):
    data_val_r = sen1_IW1_r_bytes[ii: ii + 4]
    data_val_f_r = struct.unpack('f', data_val_r)
    data_r.append(data_val_f_r[0])
    
    data_val_i = sen1_IW1_i_bytes[ii: ii + 4]
    data_val_f_i = struct.unpack('f', data_val_i)
    data_i.append(data_val_f_i[0])

data_r = np.array(data_r)
data_r = data_r.reshape([lines, samples])

data_i = np.array(data_i)
data_i = data_i.reshape([lines, samples])
#%%
inten = data_r**2 + data_i**2
plt.figure()
plt.imshow(inten, vmin = 30, vmax = 110)
#%%
save_folder = sen_folder + '\\python_generated\\split'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
io.imsave(save_folder + '\\i_' + swath_sel.upper() + '_Pol_' + pol_sel.upper() + '_split.tiff', data_r)
io.imsave(save_folder + '\\q_' + swath_sel.upper() + '_Pol_' + pol_sel.upper() + '_split.tiff', data_i)
io.imsave(save_folder + '\\intensity_' + swath_sel.upper() + '_Pol_' + pol_sel.upper() + '_split.tiff', inten)