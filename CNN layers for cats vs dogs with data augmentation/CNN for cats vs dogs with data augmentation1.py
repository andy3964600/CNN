# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:37:46 2019

@author: andy3
"""
######################################################
#This example we take datasets including 4000 pictures(2000 cats and 2000 dogs)
#
#We take 2000 pictures as training sets,1000 pictures as validation sets, 1000 pictures as testing sets.
#
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d_7 (Conv2D)            (None, 148, 148, 32)      896       
#_________________________________________________________________
#max_pooling2d_7 (MaxPooling2 (None, 74, 74, 32)        0         
#_________________________________________________________________
#conv2d_8 (Conv2D)            (None, 72, 72, 64)        18496     
#_________________________________________________________________
#max_pooling2d_8 (MaxPooling2 (None, 36, 36, 64)        0         
#_________________________________________________________________
#conv2d_9 (Conv2D)            (None, 34, 34, 128)       73856     
#_________________________________________________________________
#max_pooling2d_9 (MaxPooling2 (None, 17, 17, 128)       0         
#_________________________________________________________________
#conv2d_10 (Conv2D)           (None, 15, 15, 128)       147584    
#_________________________________________________________________
#max_pooling2d_10 (MaxPooling (None, 7, 7, 128)         0         
#_________________________________________________________________
#Dropout 50 percent           (None, 7, 7, 128)         0         
#_________________________________________________________________
#flatten_3 (Flatten)          (None, 6272)              0         
#_________________________________________________________________
#dense_5 (Dense)              (None, 512)               3211776   
#_________________________________________________________________
#dense_6 (Dense)              (None, 1)                 513       
#=================================================================
#Total params: 3,453,121
#Trainable params: 3,453,121
#Non-trainable params: 0
#_________________________________________________________________
#
#
#
#
#
######################################################

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

validation_cats_dir=os.path.join(validation_dir,'cats')
if not os.path.isdir(validation_cats_dir):os.mkdir(validation_cats_dir)

validation_dogs_dir=os.path.join(validation_dir,'dogs')
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
    
    st=os.path.join(train_cats_dir,
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

#training,validation,testing of dogs

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
##################################################################
#
#Create the CNN layers
#
##################################################################
from keras import layers

from keras import models

#Create our CNN layers

model=models.Sequential()

#1 Conv2D layer as input layer with 32 neural network union

model.add(layers.Conv2D(32,
                        (3,3),
                        activation='relu',
                        input_shape=(150,150,3)))

#1 MaxPooling2D layer

model.add(layers.MaxPool2D((2,2)))

#2 Conv2D layer with 64 neural network union

model.add(layers.Conv2D(64,
                        (3,3),
                        activation='relu'))

#2 MaxPooling2D layer

model.add(layers.MaxPool2D((2,2)))

#3 Conv2D layer with 64 neural network union

model.add(layers.Conv2D(128,
                        (3,3),
                        activation='relu'))

#3 MaxPooling2D layer

model.add(layers.MaxPool2D((2,2)))

#4 Conv2D layer with 64 neural network union

model.add(layers.Conv2D(128,
                        (3,3),
                        activation='relu'))

#4 MaxPooling2D layer

model.add(layers.MaxPool2D((2,2)))

#We flat the 3D tensor to 1D array.

model.add(layers.Flatten())

#1D layers with Dropout 50 percent neural network union.

model.add(layers.Dropout(0.5))

#1D layers with 512 neural network union.

model.add(layers.Dense(512,
                       activation='relu'))

#the output layers(softmax) with 10 NN union.

model.add(layers.Dense(1,
                       activation='sigmoid'))

#the strusture of CNN.

model.summary()

###################################################################
#
#
#
#Create model to train our CNN
#
#
#
###################################################################
from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

###################################################################
##############Data augmentation method#############################
#
#Because our data are JPEG file, we need to trans JPEG to RGB tensor,The progress can be given by:
#
#1.load the JPEG file
#
#2.Decode the JPEG to RGB
#
#3.Let RGB trans to (0~255)tensor
#
#4.Normalize the range(0~255) between [0,1]
#
#Keras has the package of ImageDataGenerator type, the ImageDataGenerator can automatically trans JPEG to tensor.
#
##################################################################
#The setup can be written as:

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,
                                 rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                )

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(train_dir,
                                                  target_size=(150,150),
                                                  batch_size=32,
                                                  class_mode='binary')

validation_generator=test_datagen.flow_from_directory(validation_dir,
                                                      target_size=(150,150),
                                                      batch_size=32,
                                                      class_mode='binary')
#Save the model(CNN)

model.save('cats_and_dogs_small_2.h5')

#Show the loss quantity and accuracy quantity of training(validation) period.

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
