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


# path = 'C:/Users/Anonymous/Desktop/flower_photos'
path = '/home/unagi/IndianFoodRecognition/oneShotLearning/Data'
predCSV = '/home/unagi/IndianFoodRecognition/oneShotLearning//predictions.csv'

def W_init(shape, name=None):
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return backend.variable(values, name=name)

def b_init(shape, name=None):
    values=rng.normal(loc=0.5, scale=1e-2, size=shape)
    return backend.variable(values, name=name)

def predict(PredPath,siamese_network):
    generalSelections = []
    predictionImage = []
    labels = []
    categories = 0
    for subdir,dir,files in os.walk(path):
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
    preds = siamese_network.predict({'left_input': np.array(generalSelections),'right_input': np.array(predictionImage)})
    for i in range(len(preds)):
        print(labels[i]+' ---> '+str(preds[i]))


dataset = util.dataset_loader(path, (128, 128, 1))
left_inputs, right_inputs, labels = dataset.get_dataset()

img_shape = (128, 128, 1)
left_input = Input(img_shape, name='left_input')
right_input = Input(img_shape, name='right_input')

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(2e-4), kernel_initializer=W_init, bias_initializer=b_init, input_shape=[128,128,1]))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Conv2D(filters=20, kernel_size=5, padding='same', kernel_regularizer=l2(2e-4), kernel_initializer=W_init, bias_initializer=b_init, activation='relu'))
model.add(MaxPool2D(pool_size=(4, 4)))
model.add(BatchNormalization())
model.add(Conv2D(filters=40, kernel_size=5, padding='same', kernel_regularizer=l2(2e-4),  kernel_initializer=W_init, bias_initializer=b_init, activation='relu'))
model.add(MaxPool2D(pool_size=(4 ,4)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(320, activation='sigmoid', kernel_regularizer=l2(1e-3),  kernel_initializer=W_init, bias_initializer=b_init))

encoding_left = model(left_input)
encoding_right = model(right_input)

distance = lambda x : backend.abs((x[0] ** 2) - (x[1] ** 2))
merged_vector = merge(inputs=[encoding_left, encoding_right], mode=distance, output_shape=lambda x: x[0])
predict_layer = Dense(1, name='main_output')(merged_vector)
siamese_network = Model(inputs=[left_input, right_input], outputs=predict_layer)
optimizer = Adam(0.00006)

siamese_network.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

siamese_network.fit({'left_input': left_inputs, 'right_input': right_inputs}, {'main_output': labels}, epochs=5, verbose=1, validation_split=0.2)

predict('/home/unagi/IndianFoodRecognition/oneShotLearning/aloo-baingan.jpg',siamese_network)

if os.path.exists('model.json'):
    os.remove('model.json')
    os.remove('model.h5')
model_json = siamese_network.to_json()
with open('model.json','w') as json_file:
    json_file.write(model_json)
siamese_network.save_weights('model.h5')
print('model saved to disk')