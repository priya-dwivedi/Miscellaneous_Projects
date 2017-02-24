#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:32:35 2017

@author: priyankadwivedi
"""

## Load and check image sizes
import glob
import os
from PIL import Image
import csv
import numpy as np
import cv2
import sklearn

# TODO: Add path where the train images are unzipped and stored
#paths = "/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/data/"
#os.chdir(r"/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/data/")
#paths = "/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/data_combined"
#os.chdir(r"/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/data_combined")
paths = "/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/my_driving/center_curves"
os.chdir(r"/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/my_driving/center_curves")
new_path = os.path.join(paths, "IMG/", "*.jpg")
cwd = os.getcwd()
print(cwd)

for infile in glob.glob(new_path)[:2]:
    im = Image.open(infile)
    print(im.size, im.mode)

# All images are 320x160
# Cut size by half and overwrite existing image

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print(train_samples[0])

include = False
def generator(samples, batch_size=32, include = include):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        from sklearn.utils import shuffle
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            if include == True:
                for batch_sample in batch_samples:
                    center_name = './IMG/'+batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(center_name)
                    left_name = './IMG/'+batch_sample[1].split('/')[-1]
                    left_image = cv2.imread(left_name)
                    right_name = './IMG/'+batch_sample[2].split('/')[-1]
                    right_image = cv2.imread(right_name)

                    center_angle = float(batch_sample[3])

                    correction = 0.05
                    left_angle = center_angle + correction
                    right_angle = center_angle - correction
                    images.extend([center_image, left_image, right_image])
                    angles.extend([center_angle, left_angle, right_angle])
            else:
                for batch_sample in batch_samples:
                    center_name = './IMG/'+batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(center_name)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

if include == True:
    samples_per_epoch = len(train_samples)*3
    nb_val_samples = len(validation_samples)*3
else:
    samples_per_epoch = len(train_samples)
    nb_val_samples = len(validation_samples)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, RMSprop



#Params
row, col, ch = 160, 320, 3

batch_size = 32
nb_epoch = 8
nb_classes = 1


def create_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Lambda(lambda x: (x / 127.5 - 1.)))
    # Step 1: Convolution Layer with patch size = 5, stride = 1, same padding an depth = 24
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    #model.compile(optimizer="adam", loss="mse")

    return model

model = create_model()
print(len(model.layers))

weights_path = '/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/model1.h5'
model.load_weights(weights_path)


for layer in model.layers[:14]:
    layer.trainable = False

model.compile(loss='mse',
              optimizer=Adam(lr= 0.0005))

history_object = model.fit_generator(train_generator, samples_per_epoch= samples_per_epoch,
                                     validation_data=validation_generator,
                                     nb_val_samples=nb_val_samples, nb_epoch=nb_epoch, verbose=1)

import matplotlib.pyplot as plt

print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

from keras.models import model_from_json

model_json = model.to_json()
with open("/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/model3.json", "w") as json_file:
    json_file.write(model_json)

model.save("/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/model2.h5")
print("Saved model to disk")

print(model.summary())

# # # list all data in history

# # this is the augmentation configuration we will use for training
# # train_datagen = ImageDataGenerator(
# #        rescale=1./255,
# #        shear_range=0.2,
# #        zoom_range=0.2,
# #        horizontal_flip=True)
