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
def W_init(shape, name=None):
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return backend.variable(values, name=name)

def b_init(shape, name=None):
    values=rng.normal(loc=0.5, scale=1e-2, size=shape)
    return backend.variable(values, name=name)

def loadModel():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

def checkCorrectness(PredPath,CatPath):
    if not os.path.exists('feedbackQ.json'):
        with open('feedbackQ.json','w') as outfile:
            json.dump({},outfile)
    queue = open('feedbackQ.json','r').read()
    loaded_json = json.loads(queue)
    if len(loaded_json) == 10:
        for x in loaded_json:
            os.rename(x,loaded_json[x])
        exec('main.py')
        os.remove('feedbackQ.json')
        with open('feedbackQ.json','w') as outfile:
            json.dump({},outfile)
    correct = input('Is the prediction correct? (Y/N):')
    if correct=='N':
        actualLabel = input('What is the name of the dish? :')
        loaded_json[PredPath] = CatPath+'/'+str(uuid.uuid4())+'.'+PredPath.split('.')[-1]
        with open('feedbackQ.json','w') as outfile:
            json.dump(loaded_json,outfile)



def predict(PredPath):
    img_shape = (128,128,1)
    CatPath = '/home/unagi/IndianFoodRecognition/oneShotLearning/Data'
    siamese_network = loadModel()
    generalSelections = []
    predictionImage = []
    labels = []
    categories = 0
    for subdir,dir,files in os.walk(CatPath):
        categories += len(dir)
        if len(files)!=0:
            imgPath = os.path.join(subdir,files[0])
            labels.append(subdir.split('/')[-1])
            img = misc.imread(imgPath,'L')
            img = np.reshape(misc.imresize(img, img_shape), img_shape)
            generalSelections.append(img)
    predImg = misc.imread(PredPath,'L')
    predImg = np.reshape(misc.imresize(predImg, img_shape), img_shape)
    predictionImage = [predImg] * categories
    # siamese_network.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    preds = siamese_network.predict({'left_input': np.array(generalSelections),'right_input': np.array(predictionImage)})
    predDict = {}
    for i in range(len(preds)):
        predDict[labels[i]] = float(preds[i])
    with open('predictions.json','w') as outfile:
        json.dump(predDict, outfile)
    for k,v in sorted(predDict.items()):
        print(k,v)
    checkCorrectness(PredPath,CatPath)

predict('/home/unagi/IndianFoodRecognition/oneShotLearning/Test/aloo_matar.jpg')
