import numpy as np
import pandas as pd
import math
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, LeakyReLU
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.layers.normalization import BatchNormalization
import sys


def add_padding(feature, label):
    new_feature = np.empty((len(feature)*6, 48, 48, 1))
    new_label = np.empty(len(label)*6)
    for idx in range(0, len(feature)):
        temp = feature[idx]
        i = 6*idx
        new_feature[i] = np.pad(temp[0:42,0:42,:], ((3,3),(3,3),(0,0)), 'constant')
        new_feature[i+1] = np.pad(temp[6:48,0:42,:], ((3,3),(3,3),(0,0)), 'constant')
        new_feature[i+2] = np.pad(temp[6:48,6:48,:], ((3,3),(3,3),(0,0)), 'constant')
        new_feature[i+3] = np.pad(temp[0:42,6:48,:], ((3,3),(3,3),(0,0)), 'constant')
        new_feature[i+4] = np.pad(temp[3:45,3:45,:], ((3,3),(3,3),(0,0)), 'constant')
        new_feature[i+5] = temp
        new_label[i:i+6] = label[idx]
    return new_feature, new_label

def add_fliplr(feature, label):
    new_feature = np.empty((len(feature)*2, 48, 48, 1))
    new_label = np.empty(len(label)*2)
    for idx in range(0, len(feature)):
        temp = feature[idx]
        i = 2*idx
        new_feature[i] = temp
        new_feature[i+1] = np.fliplr(temp)
        new_label[i:i+2] = label[idx]
    return new_feature, new_label

def data_augmentation(feature, label):
    feature, label = add_padding(feature, label)
    feature, label = add_fliplr(feature, label)
    return feature, label



# read train data
df = pd.read_csv(sys.argv[1])
label = np.array(df['label'].values, dtype='float64')             
feature_str = np.array(df['feature'].values)
size = int(math.sqrt(len(feature_str[0].split())))
length = len(feature_str)
feature = np.empty((length, size*size))
for i in range(0, length):
    feature[i] = (np.array(feature_str[i].split(),dtype = 'float64'))/255
feature = feature.reshape(-1 , size, size, 1)
tra_x = feature[4000:]
tra_y = label[4000:]
val_x = feature[:4000]
val_y = label[:4000]

tra_x,tra_y = data_augmentation(tra_x,tra_y)

tra_y = np_utils.to_categorical(tra_y,7)
val_y = np_utils.to_categorical(val_y,7)



# construct model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(units = 256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 7, activation='softmax'))
model.summary()
# sgd = SGD(lr=0.005, decay=0.00001, momentum=0.9)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])



# train preocess
filep="weights-improvement-{epoch:04d}-{val_acc:.5f}.h5"
checkpointer = ModelCheckpoint(monitor='val_acc', filepath=filep, verbose=1, save_best_only=True)
train_history = model.fit(x=tra_x, y=tra_y, validation_data=(val_x, val_y), epochs=150, batch_size=50, callbacks=[checkpointer])  
model.save('best_model.h5')