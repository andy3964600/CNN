# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:48:44 2019

@author: andy3
"""

#First,we need to download the file on kaggle"cat-vs-dogs"
import os,shutil

original_dataset_dir=r'C:\Users\andy3\train'

base_dir=r"C:\Users\andy3\cats_and_dogs_small"

if not os.path.isdir(base_dir):os.mkdir(base_dir)

train_dir=os.path.join(base_dir,
                       'train')
if not os.path.isdir(train_dir):os.mkdir(train_dir)

validation_dir=os.path.join(base_dir,
                            'validation')
if not os.path.isdir(validation_dir):os.mkdir(validation_dir)

test_dir=os.path.join(base_dir,
                      'test')
if not os.path.isdir(test_dir):os.mkdir(test_dir)

train_cats_dir=os.path.join(train_dir,
                            'cats')
if not os.path.isdir(train_cats_dir):os.mkdir(train_cats_dir)

train_dogs_dir=os.path.join(train_dir,
                            'dogs')
if not os.path.isdir(train_dogs_dir):os.mkdir(train_dogs_dir)

validation_cats_dir=os.path.join(validation_dir,
                                 'cats')
if not os.path.isdir(validation_cats_dir):os.mkdir(validation_cats_dir)

validation_dogs_dir=os.path.join(validation_dir,
                                 'dogs')
if not os.path.isdir(validation_dogs_dir):os.mkdir(validation_dogs_dir)

test_cats_dir=os.path.join(test_dir,
                           'cats')
if not os.path.isdir(test_cats_dir):os.mkdir(test_cats_dir)

test_dogs_dir=os.path.join(test_dir,
                           'dogs')
if not os.path.isdir(test_dogs_dir):os.mkdir(test_dogs_dir)

#Create the training sets , validation sets ,testing sets of cats

fnames=['cat.{}.jpg'.format(i) for i in range(1000)]

for fname in fnames:
    
    src=os.path.join(original_dataset_dir,
                     fname)
    
    dst=os.path.join(train_cats_dir,
                     fname)
    
    shutil.copyfile(src,
                    dst)

fnames=fnames=['cat.{}.jpg'.format(i) for i in range(1000,1500)]

for fname in fnames:
    
    src=os.path.join(original_dataset_dir,
                     fname)
    
    dst=os.path.join(validation_cats_dir,
                     fname)
    
    shutil.copyfile(src,
                    dst)

fnames=fnames=['cat.{}.jpg'.format(i) for i in range(1500,2000)]

for fname in fnames:
    
    src=os.path.join(original_dataset_dir,
                     fname)
    
    dst=os.path.join(test_cats_dir,
                     fname)
    
    shutil.copyfile(src,
                    dst)

#Create the training sets , validation sets ,testing sets of dogs

fnames=['dog.{}.jpg'.format(i) for i in range(1000)]

for fname in fnames:
    
    src=os.path.join(original_dataset_dir,
                     fname)
    
    dst=os.path.join(train_dogs_dir,
                     fname)
    
    shutil.copyfile(src,
                    dst)

fnames=['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    
    src=os.path.join(original_dataset_dir,
                     fname)
    
    dst=os.path.join(validation_dogs_dir,
                     fname)
    
    shutil.copyfile(src,
                    dst)

fnames=['dog.{}.jpg'.format(i) for i in range(1500,2000)]

for fname in fnames:
    
    src=os.path.join(original_dataset_dir,
                     fname)
    
    dst=os.path.join(test_dogs_dir,
                     fname)
    
    shutil.copyfile(src,
                    dst)
####################################################
#
#
#
#
##################The method 2##################################
#################Feature extraction from VGG16#################
#
#
#
#
#
#
####################################################
#Import the trained CNN ->VGG16
from keras.applications import VGG16

from keras import models

from keras import layers

conv_base=VGG16(weights='imagenet',
                include_top=False,
                input_shape=(150,150,3))
#The structure of CNN layers can be builded as:

model=models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(256,
                       activation='relu'))

model.add(layers.Dense(1,
                       activation='sigmoid'))

model.summary()
######################################################
#
#
#Next, we need to freeze the weights of VGG16,Because we don't want to change its weight.
#
#
######################################################
#Freeze the layers of VGG16.
######################################################

#Before freezing the conv_base

print('this is the number of trainable weights'
      'before freezing the conv_base:',len(model.trainable_weights))

#After freezing the conv_base

conv_base.trainable=False

print('this is the number of trainable weights'
      'before freezing the conv_base:',len(model.trainable_weights))

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

train_datagen=ImageDataGenerator(rescale=1./255,
                                 rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(train_dir,
                                                  target_size=(150,150),
                                                  batch_size=32,
                                                  class_mode='binary')

validation_generator=test_datagen.flow_from_directory(validation_dir,
                                                      target_size=(150,150),
                                                      batch_size=32,
                                                      class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history=model.fit_generator(train_generator,
                            steps_per_epoch=100,
                            epochs=30,
                            validation_data=validation_generator,
                            validation_steps=50)

import matplotlib.pyplot as plt

acc=history.history['acc']

val_acc=history.history['val_acc']

loss=history.history['loss']

val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,
         acc,
         'bo',
         label='Traning accuracy')

plt.plot(epochs,
         val_acc,
         'b',
         label='Validation accuracy')

plt.title('The accuracy of training and validation')

plt.legend()

plt.figure()

plt.plot(epochs,
         loss,
         'bo',
         label='Training loss quantity')

plt.plot(epochs,
         val_loss,
         'b',
         label='Validation loss quantity')

plt.title('Training and validation loss')

plt.legend()

plt.show()


