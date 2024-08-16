import cv2
from PIL import ImageTk, Image


import numpy as np
import time
import glob
import os
import shutil
import pickle



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,Activation,SpatialDropout2D
from tensorflow.keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing import image
import keras

import matplotlib.pyplot as plt





file_path='D:/pythoncode/test/1.jpg'
dir_path = 'D:/pythoncode/train'
img_list = glob.glob(os.path.join(dir_path, '*/*.jpg'))
train=ImageDataGenerator(horizontal_flip=True,vertical_flip=True,validation_split=0.1,rescale=1./255,shear_range = 0.1,zoom_range = 0.1,width_shift_range = 0.1,height_shift_range = 0.1,)
train_generator=train.flow_from_directory(dir_path,target_size=(100,100),batch_size=32,class_mode='categorical',subset='training')
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

model = load_model('trained_model.h5')
img = keras.utils.load_img(file_path, target_size=(100, 100))
img = keras.utils.img_to_array(img, dtype=np.uint8)
img=np.array(img)/255.0
p=model.predict(img[np.newaxis, ...],verbose=0)
predicted_class = labels[np.argmax(p[0], axis=-1)]
print("Classified:",predicted_class)

