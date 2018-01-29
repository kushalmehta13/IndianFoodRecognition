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


# path = 'C:/Users/Anonymous/Desktop/flower_photos'
path = '/home/unagi/IndianFoodRecognition/oneShotLearning/Data'
predCSV = '/home/unagi/IndianFoodRecognition/oneShotLearning//predictions.csv'

def W_init(shape, name=None):
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return backend.variable(values, name=name)

def b_init(shape, name=None):
    values=rng.normal(loc=0.5, scale=1e-2, size=shape)
    return backend.variable(values, name=name)

dataset = util.dataset_loader(path, (128, 128, 1))
left_inputs, right_inputs, labels = dataset.get_dataset()

img_shape = (128, 128, 1)
left_input = Input(img_shape, name='left_input')
right_input = Input(img_shape, name='right_input')

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

distance = lambda x : K.abs((x[0] ** 2) - (x[1] ** 2))
# merged_vector = Lambda(lambda x : K.abs((x[0] ** 2) - (x[1] ** 2)))([encoding_left,encoding_right])
merged_vector = merge(inputs=[encoding_left, encoding_right], mode=distance, output_shape=lambda x: x[0])
# encoding_left_squared = Multiply()([encoding_left,encoding_left])
# encoding_right_squared = Multiply()([encoding_right,encoding_right])
# encoding_subtracted = Subtract()([encoding_left_squared,encoding_right_squared])
# merged_vector = K.abs(encoding_subtracted)
predict_layer = Dense(1, name='main_output')(merged_vector)
siamese_network = Model(inputs=[left_input, right_input], outputs=predict_layer)
optimizer = Adam(0.00006)

siamese_network.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

siamese_network.fit({'left_input': left_inputs, 'right_input': right_inputs}, {'main_output': labels}, epochs=10, verbose=1, validation_split=0.2)



if os.path.exists('model.h5'):
    os.remove('model.json')
    os.remove('model.h5')
model_json = siamese_network.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
siamese_network.save_weights("model.h5")
print("Saved model to disk")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
