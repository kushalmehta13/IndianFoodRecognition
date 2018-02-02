import util
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dense, Flatten, merge
from keras.optimizers import Adam
from keras import Sequential
from keras import Model
from keras.models import model_from_json
from keras import Input
from keras import backend
from keras.regularizers import l2
import cv2
import csv
import numpy as np
from scipy import misc
import numpy.random as rng
import os
import uuid
import json
import operator

def loadModel():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

def predict(PredPath):
    img_shape = (128,128,1)
    CatPath = '/home/unagi/IndianFoodRecognition/oneShotLearning/Dataset/Food Images/Train'
    siamese_network = loadModel()
    generalSelections = []
    predictionImage = []
    labels = []
    categories = 0
    for subdir,dir,files in os.walk(CatPath):
        categories += len(dir)
        if len(files)>2:
            imgPath = os.path.join(subdir,files[0])
            labels.append(subdir.split('/')[-1])
            img = misc.imread(imgPath,'L')
            img = np.reshape(misc.imresize(img, img_shape), img_shape)
            generalSelections.append(img)
    predDict = {}
    for subdir,dir,files in os.walk(PredPath):
        if 'test' in subdir:
            continue
        predCat = subdir.split('/')[-1]
        predDict[predCat] = {}
        if len(files)>0:
            for file in files:
                imgPath = os.path.join(subdir,file)
                predImg = misc.imread(imgPath,'L')
                predImg = np.reshape(misc.imresize(predImg, img_shape), img_shape)
                predictionImage = [predImg] * categories
                preds = siamese_network.predict({'left_input': np.array(generalSelections),'right_input': np.array(predictionImage)})
                predDict[predCat][file] = {}
                for i in range(len(preds)):
                    predDict[predCat][file][labels[i]] = float(preds[i])
    with open('predictions.json','w') as outfile:
        json.dump(predDict,outfile)
predict('/home/unagi/IndianFoodRecognition/oneShotLearning/Dataset/Food Images/Test')
