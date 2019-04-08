#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 09:07:02 2019

@author: achyutajha
"""
import os
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageSequence

labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,
                   'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,
                   'Z':25,'space':26,'del':27,'nothing':28}

im = Image.open("demo.gif")
ch = 'a'
index = 1
for frame in ImageSequence.Iterator(im):
    frame.save("./gif_images/%s.png" % ch)
    ch = chr(ord(ch) + 1) 
    
size = 64,64    
model = load_model('sign_detection.h5')

output_sentence = 'Alexa '
for image in sorted(os.listdir('./gif_images')):
    try:
        temp_img = cv2.imread('./gif_images' + '/' + image)
        temp_img = cv2.resize(temp_img, size)
        test_image = np.expand_dims(temp_img, axis = 0)
        result = model.predict(test_image)
        for letter, index in labels_dict.items():    
            if index == result.argmax():
                if letter!='space':
                    output_sentence = output_sentence + letter 
                else:
                    output_sentence = output_sentence + ' ' 
    except Exception as e:
        pass
output_sentence = output_sentence.replace('WHAT',"WHAT'S")
print(output_sentence)