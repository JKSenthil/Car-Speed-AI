from tensorflow import keras

# TODO implement model on this site: https://github.com/experiencor/speed-prediction/blob/master/README.md

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from tensorflow import keras
import keras.models as models
from keras.optimizers import SGD, Adam, RMSprop
from imgaug import augmenters as iaa
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import MaxPooling2D, UpSampling2D, Conv2D, Conv2DTranspose, ZeroPadding2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda, add, LSTM, TimeDistributed, concatenate
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.utils import conv_utils
from keras.engine.topology import Layer
from keras.engine import InputSpec
import numpy as np
from keras.layers import LSTM
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input

FRAME_H, FRAME_W = 112, 112
TIMESTEPS = 16

##### MODEL #####

model = Sequential()
input_shape=(TIMESTEPS, FRAME_H, FRAME_W, 3) # l, h, w, c

# 1st layer group
model.add(Conv3D(64, (3, 3, 3),  activation='relu', padding='same', name='conv1', input_shape=input_shape))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1'))

# 2nd layer group
model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2'))

# 3rd layer group
model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3a'))
model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3b'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3'))

# 4th layer group
model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4a'))
model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4b'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4'))

# 5th layer group
model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5a'))
model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5b'))
model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5'))
model.add(Flatten())

# FC layers group
model.add(Dense(4096, activation='relu', name='fc6'))
model.add(Dropout(.5))
model.add(Dense(4096, activation='relu', name='fc7'))
model.add(Dropout(.5))
model.add(Dense(1,    activation='linear', name='fc8'))

model.summary()



##### TRAINING #####

DATA_PATH = "../input/train.txt"
TRAIN_VIDEO = os.path.join(DATA_PATH, '../input/train.mp4')

train_frames = 20400

video_reader = cv2.VideoCapture(TRAIN_VIDEO)
train_image = 'data/' + video_file + '/images/'
train_label = open(DATA_PATH, 'w')

counter = 0

# Number Frames ?
while(True):
    ret, frame = video_reader.read()

    if ret == True:
        cv2.imwrite(train_image + str(counter).zfill(6) + '.png', frame)
        train_label.write(label_inp[counter])
        counter += 1
    else:
        break

video_reader.release()
train_label.close()


##### TEST #####


early_stop  = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2, mode='min', verbose=1)
checkpoint  = ModelCheckpoint('weight_c3d.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)

data_folder = 'data/train/' # NOT RIGHT
split_ratio = 0.90

indices = range(TIMESTEPS-1, len(os.listdir(data_folder + 'images/')), TIMESTEPS)
np.random.shuffle(indices)

train_indices = list(indices[0:int(len(indices)*split_ratio)])
valid_indices = list(indices[int(len(indices)*split_ratio):])

gen_train = BatchGenerator(data_folder, train_indices, batch_size=4, timesteps=TIMESTEPS)
gen_valid = BatchGenerator(data_folder, valid_indices, batch_size=4, timesteps=TIMESTEPS, jitter = False)

def custom_loss(y_true, y_pred):
    loss = tf.squared_difference(y_true, y_pred)
    loss = tf.reduce_mean(loss)
    
    return loss
