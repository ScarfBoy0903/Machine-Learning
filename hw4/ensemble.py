from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, LeakyReLU
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd
import math

import csv
import sys

model1 = load_model('ADAM_best.h5')
model2 = load_model('SGD_best.hdf5')
model3 = load_model('SGD_best2.h5')
model4 = load_model('ADAM_best2.h5')

import keras.layers
input1 = keras.layers.Input(shape=(48,48,1))
x1 = model1(input1)
x2 = model2(input1)
input2 = keras.layers.Input(shape=(48,48,1))  # if two models use the same input, only need input1, and change the input of model2 to input1
x3 = model3(input2)
x4 = model4(input2)

out = keras.layers.Average()([x1, x2, x3, x4])
New_model = keras.models.Model(inputs=[input1, input2], outputs=out)

sgd = SGD(lr=0.005, decay=0.00001, momentum=0.9, nesterov=False)
New_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

test_df = pd.read_csv(sys.argv[1])
test_str = np.array(test_df['feature'].values)
size = int(math.sqrt(len(test_str[0].split())))
length = len(test_str)
test_data = np.empty((length, size*size))
test_data_128 = np.empty((length, size*size))

for i in range(0, len(test_str)):
    test_data[i] = np.array(test_str[i].split(),dtype = 'float64')/255
    test_data_128[i] = (np.array(test_str[i].split(),dtype = 'float64')-127.5)/255
test_data = test_data.reshape(-1, size, size, 1)
test_data_128 = test_data_128.reshape(-1, size, size, 1)

pre_label = New_model.predict([test_data,test_data_128])

ans = np.empty(len(pre_label))
for i in range(len(pre_label)):
    ans[i] = pre_label[i].argmax()

# output the csv
text = open('predict.csv', "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "label"])
for i in range(len(ans)):
    s.writerow([str(i), int(ans[i])])
text.close()
