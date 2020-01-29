# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:05:49 2020

@author: lenovo
"""

import cv2
import tensorflow as tf

CATEGORIES=["Dog","Cat"]

def prepare(filepath):
    IMG_SIZE=50
    img_array=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



model=tf.keras.models.load_model("64x3-CNN.model")
predection=model.predict([prepare('input/download1.jpg')])
print(CATEGORIES[int(predection[0][0])])

