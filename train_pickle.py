# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:22:28 2019

@author: lenovo
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

DATADIR="C:/Users/lenovo/Desktop/Cat dog/PetImages"
CATEGORIES=["Dog","Cat"]

training_data=[]
IMG_SIZE = 50

def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array= cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array,class_num])
                #plt.imshow(new_array, cmap='gray')
                #plt.show()
                #print(img_array)
            except Exception as e:
                pass

create_training_data()
print(len(training_data))
random.shuffle(training_data)

X=[]
Y=[]
for features,labels in training_data:
    X.append(features)
    Y.append(labels)
X= np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
pickle_out= open('X.pickle',"wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out= open('Y.pickle',"wb")
pickle.dump(Y,pickle_out)
pickle_out.close()
    
pickle_in= open('X.pickle',"rb")
X=pickle.load(pickle_in)
print(X[1])    












           

        
    
    
