from tkinter import filedialog

import cv2
import  imutils

import numpy as np
import time
import glob
import os
import shutil
import pickle



from sklearn.model_selection import train_test_split

import keras
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,Activation,SpatialDropout2D
from tensorflow.keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing import image

import matplotlib.pyplot as plt



dir_path  = 'D:/pythoncode/train'
img_list = glob.glob(os.path.join(dir_path, '*/*.jpg'))
print(len(img_list))


train=ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         validation_split=0.1,
                         rescale=1./255,
                         shear_range = 0.1,
                         zoom_range = 0.1,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,)

test=ImageDataGenerator(rescale=1/255,
                        validation_split=0.1)

train_generator=train.flow_from_directory(dir_path,
                                          target_size=(100,100),
                                          batch_size=32,
                                          class_mode='categorical',
                                          subset='training')

test_generator=test.flow_from_directory(dir_path,
                                        target_size=(100,100),
                                        batch_size=32,
                                        class_mode='categorical',
                                        subset='validation')

labels = (train_generator.class_indices)
print(labels)

labels = dict((v,k) for k,v in labels.items())
print(labels)

for image_batch, label_batch in train_generator:
  break
image_batch.shape, label_batch.shape


print (train_generator.class_indices)

Labels = '\n'.join(sorted(train_generator.class_indices.keys()))

with open('labels.txt', 'w') as f:
  f.write(Labels)


model=Sequential()

#Convolution blocks
model.add(Conv2D(32,(3,3), padding='same',input_shape=(100,100,3),activation='relu'))
model.add(MaxPooling2D(pool_size=2)) 
model.add(Conv2D(64,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2)) 
model.add(Conv2D(32,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2)) 

#Classification layers
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(9,activation='softmax'))

filepath="trained_model.h5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint1]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.SpecificityAtSensitivity(0.5), keras.metrics.SensitivityAtSpecificity(0.5), 'accuracy'])
history = model.fit(train_generator,epochs=100,steps_per_epoch=2276//32,validation_data=test_generator,validation_steps=251//32,workers = 4,callbacks=callbacks_list)
model.save(filepath)

f = open('history.pckl', 'wb')
pickle.dump(history.history, f)
f.close()
