import util
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dense, Flatten, merge, Multiply, Subtract, concatenate, Lambda
from keras.optimizers import Adam
from keras import Sequential
from keras import Model
from keras.models import model_from_json
from keras import Input
from keras import backend as K
from keras.regularizers import l2
import cv2
import csv
import numpy as np
from scipy import misc
import numpy.random as rng
import os


path = '/home/unagi/IndianFoodRecognition/oneShotLearning/Dataset/Food Images/Train'
predCSV = '/home/unagi/IndianFoodRecognition/oneShotLearning//predictions.csv'

def W_init(shape, name=None):
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return backend.variable(values, name=name)

def b_init(shape, name=None):
    values=rng.normal(loc=0.5, scale=1e-2, size=shape)
    return backend.variable(values, name=name)

def saveModel(modelToSave):
    model_json = modelToSave.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    modelToSave.save_weights("model.h5")
    print("Saved model to disk")

dataset = util.dataset_loader(path, (128, 128, 1))
left_inputs, right_inputs, labels = dataset.get_dataset()

img_shape = (128, 128, 1)
left_input = Input(img_shape, name='left_input')
right_input = Input(img_shape, name='right_input')

if os.path.exists('model.h5'):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    optimizer = Adam(0.00006)
    loaded_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    #TODO: left input and right inputs need to come from the intermediate data file. The number of combinations will be small
    #      since the retrain_dataset.h5 will contain only the feedback image information.
    loaded_model.fit({'left_input': left_inputs, 'right_input': right_inputs}, {'main_output': labels}, epochs=10, verbose=1, validation_split=0.2)
    saveModel(loaded_model)
else:
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(2e-4), kernel_initializer='random_normal', bias_initializer='random_normal', input_shape=[128,128,1]))
    model.add(MaxPool2D())
    model.add(BatchNormalization())
    model.add(Conv2D(filters=20, kernel_size=5, padding='same', kernel_regularizer=l2(2e-4), kernel_initializer='random_normal', bias_initializer='random_normal', activation='relu'))
    model.add(MaxPool2D(pool_size=(4, 4)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=40, kernel_size=5, padding='same', kernel_regularizer=l2(2e-4),  kernel_initializer='random_normal', bias_initializer='random_normal', activation='relu'))
    model.add(MaxPool2D(pool_size=(4 ,4)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(320, activation='sigmoid', kernel_regularizer=l2(1e-3),  kernel_initializer='random_normal', bias_initializer='random_normal'))

    encoding_left = model(left_input)
    encoding_right = model(right_input)

    distance = lambda x : K.abs((x[0] - x[1])**2)
    merged_vector = merge(inputs=[encoding_left, encoding_right], mode=distance, output_shape=lambda x: x[0])
    predict_layer = Dense(1,activation='sigmoid', name='main_output')(merged_vector)
    siamese_network = Model(inputs=[left_input, right_input], outputs=predict_layer)
    optimizer = Adam(0.00006)

    siamese_network.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    siamese_network.fit({'left_input': left_inputs, 'right_input': right_inputs}, {'main_output': labels}, epochs=10, verbose=1, validation_split=0.2)
    saveModel(siamese_network)
