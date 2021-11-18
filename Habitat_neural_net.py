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

########################################################################################################


#  GENERATE MORE IMAGES FROM PHOTOS TAKEN BY ROTATING, ZOOMING ETC.

datagen = ImageDataGenerator(
        width_shift_range=0.1,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

dir_dir = '/Users/mark/QU/People_stuff/Josh/RLC_videos_Sep2020/labelled/train'

for folder_name in os.listdir(dir_dir):
    dir_dir2 = os.path.join(dir_dir, folder_name) + "/"
    print(dir_dir + '/' + folder_name)
    for filename in os.listdir(dir_dir2):
        pic = load_img(os.path.join(dir_dir2, filename))
        pic_array = img_to_array(pic)
        img = pic_array.reshape((1,) + pic_array.shape)

        train_count = 0
        for batch in datagen.flow(img, batch_size = 1,
                                  save_to_dir=dir_dir2,
                                  save_prefix='hab_2_' + str(filename[0:6]),
                                  save_format='jpeg'):
            train_count += 1
            if train_count > 1:
                break


########################################################################################################

TRAINING_DIR = "/Users/mark/QU/People_stuff/Josh/RLC_videos_Sep2020/labelled_copy/train"
training_datagen = ImageDataGenerator(
    rescale=1. / 255,
    fill_mode='nearest',
    vertical_flip=True,
    width_shift_range=0.1)

VALIDATION_DIR = "/Users/mark/QU/People_stuff/Josh/RLC_videos_Sep2020/labelled_copy/test"
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
NUM_CLASSES = 8
INPUT_SHAPE = x_train.shape[1:]

############################################################################################################

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


#   BASELINE MODEL  #

def build_model(hp):
    model = keras.models.Sequential()

    model.add(Conv2D(hp.Int("input_units", min_value=16, max_value=256, step=32), (3, 3), input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(hp.Int("n_layers", 1, 5)):
        model.add(Conv2D(hp.Int(f"conv_{i}_units", min_value=16, max_value=256, step=32), (3, 3)))
        model.add(Activation('relu'))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.1, default=0.005, step=0.01)))

    model.add(Dense(NUM_CLASSES))
    model.add(Activation("softmax"))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model

tuner = kt.Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=50,
    factor=3,
    directory=LOG_DIR
)

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)

tuner.search(x=x_train,
             y=y_train,
             epochs=50,
             batch_size=8,
             validation_data=(x_test, y_test),
             callbacks=[ClearTrainingOutput()])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('input_units')}, the best number of layers is {best_hps.get('n_layers')}, 
the best dropout is {best_hps.get('dropout')}, and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

model = tuner.hypermodel.build(best_hps)
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))


############################################################################################################

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
    tf.keras.layers.Dense(8, activation='softmax')
])

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
model.fit(train_generator, epochs = 8, validation_data = validation_generator, verbose = 1)

model.save('habitat_base_model')


