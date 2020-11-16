from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
from PIL import ImageFilter
from matplotlib import pyplot as plt
import random
import numpy as np
from random import sample
import cv2
import numpy as np
from skimage import io
from skimage.transform import rotate, AffineTransform, warp
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import os
from skimage import util
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import random
import time

import shutil
from os import path


from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import time
import pickle
import kerastuner as kt
import IPython

from keras import activations
import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from matplotlib import pyplot



#test_dir = '/Users/mark/QU/People_stuff/Josh/RLC_videos_Sep2020/density/train'
#train_dir = '/Users/mark/QU/People_stuff/Josh/RLC_videos_Sep2020/density/test'


train_dir = '/Users/mark/QU/People_stuff/Josh/RLC_videos_Sep2020/density/train'
test_dir = '/Users/mark/QU/People_stuff/Josh/RLC_videos_Sep2020/density/test'
for dirname in os.listdir(train_dir):
    from_dir = os.path.join(train_dir + '/' + dirname + '/')
    dest_dir = os.path.join(test_dir + '/' + dirname + '/')
    for filename in os.listdir(from_dir)[::5]:
        shutil.move(from_dir + filename, dest_dir)


TRAINING_DIR = "/Users/mark/QU/People_stuff/Josh/RLC_videos_Sep2020/density/train"
training_datagen = ImageDataGenerator(
    rescale=1. / 255,
    fill_mode='nearest',
    vertical_flip=True,
    width_shift_range=0.1)

VALIDATION_DIR = "/Users/mark/QU/People_stuff/Josh/RLC_videos_Sep2020/density/test"
validation_datagen = ImageDataGenerator(rescale=1. / 255,
                                        fill_mode='nearest')

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    color_mode='grayscale',
    target_size=(150, 150),
    class_mode='categorical')

x_train, y_train = next(train_generator)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    color_mode='grayscale',
    target_size=(150, 150),
    class_mode='categorical')

x_test, y_test = next(validation_generator)

LOG_DIR = f"{int(time.time())}"
NUM_CLASSES = 2
INPUT_SHAPE = x_train.shape[1:]


model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=INPUT_SHAPE),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # The forth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The fifth convolution
#    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
#    tf.keras.layers.MaxPooling2D(2, 2),

    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),

    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
model.fit(train_generator, epochs = 8, validation_data = validation_generator, verbose = 1)

model.save('density_base_model')

