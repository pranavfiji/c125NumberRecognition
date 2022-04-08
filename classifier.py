from matplotlib import number
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
# C:/Users/12029/venv/Scripts/python c:/Users/12029/Documents/WhiteHatProjects/DuringClassProjectd/c123/digitRecogniton.py
from sklearn.model_selection import train_test_split as tts 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as ass
import os,ssl,time 
import cv2  

X,y = fetch_openml("mnisht_784",version=1,return_X_y=True)

xTrain,xTest,yTest,yTrain=tts(X,y,train_size=1000,test_size=3000,random_state=9)

xTrainScaled=xTrain/255
xTestScale=xTest/255

model=LogisticRegression(solver="saga",multi_class="multinomial").fit(xTrainScaled,yTrain)

def getPrediction(number):

    iampill=number.open(number)
    numberBW = number.open(number)
    numberBW = iampill.convert("L")
    #this number will be represented by a single from 0 to 255
    numberBWResized=numberBW.resize((28,28),number.ANTIALIAS)
    numberInverted=PIL.numberOps.invert(numberBWResized)
    pixelFilter=20
    minPixel=np.percentile(numberInverted,pixelFilter)

    numberBwInvertedScale = np.clip(numberBWResized - minPixel,0,255)
    maxPixel = np.max(numberBWResized)
    numberBwInvertedScale = np.asarray(numberBwInvertedScale)/maxPixel
    testSample = np.array(numberBwInvertedScale).reshape(1,784)
    testPredict=model.predict(testSample)
    return testPredict[0]   
