from charloc3 import identify

import numpy as np
import random

import keras
from keras import datasets, layers, models, preprocessing

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

import json
import os

# get plate cropping model
json_file = open('model_v3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
crop_plate = models.model_from_json(loaded_model_json)
crop_plate.load_weights("model_v3_3.h5")

# get char detect model
json_file = open('charcollectedvgg.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
char_detect = models.model_from_json(loaded_model_json)
char_detect.load_weights("charcollectedvgg.h5")

def detectplate(img_path):

    # prediction of the bounding box
    im = Image.open(img_path)
    im.thumbnail((400, 300))
    ini = np.array([np.asarray(im)]) / 255
    out = crop_plate.predict(ini)[0]

    # coords of bounding box            
    x1, y1, x2, y2= out[0] * im.size[0], out[1] * im.size[1], out[2] * im.size[0], out[3] * im.size[1] 
    im_h = abs(y1 - y2)
    y1 -= im_h; y2 += im_h

    # cropping image (top left, bottom right)
    cropped = im.crop((x1 - 10, y1 - 10, x2 + 10, y2 + 10))

    # convert the cropped plate thingy into a open cv thingy
    open_cv_image = np.array(cropped) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    # get the box of each char
    lis = identify(open_cv_image)    

    res = ''
    # detect individual char
    for coord in lis:
        crop_char = cropped.crop((coord[0][0], coord[0][1], coord[1][0], coord[1][1]))
        crop_char = crop_char.resize((20, 30))
        # get prediction
        ini = np.array([np.asarray(crop_char)]) / 255
        out = char_detect.predict(ini)[0]
        # identify prediction
        dicta = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
        maxi = 0
        idx = 0
        for i in range(0, len(out)):
            if i == 0:
                maxi = out[i]
            elif(out[i] > maxi):
                maxi = out[i]
                idx = i
        res = res + dicta[idx]
        
    return res