# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:50:46 2019

@author: andy3
"""

###############################################################
#
#Use pre-training of CNN,and we import 'VGG16'
#
#pre-CNN has two method,one is feature extraction , another is fine-tunimg.
#
#In this CNN,we use 'feature extraction'.
#
#
#
###############################################################
from keras.applications import VGG16
###############################################################
#
#'weights' were used to initialize the checking points of weight of model.
#
#'include_top' is classifier of layers,in this case,we don't use this classifier.
#
###############################################################
conv_base=VGG16(weights='imagenet',
                include_top=False,
                input_shape=(150,150,3))

conv_base.summary()
###############3###############################################
#
#Next, we have two different method to solve the questions.
#
#1.We can take the feature extraction of VGG16(putting in numpy array) and iuput our dense layers.
#
#2.We can add the Dense layers to train our CNN including VGG16.
#
#
#
#
###############################################################

#################The method 1##################################
#################Feature extraction from VGG16#################
import os

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

base_dir=r"C:\Users\andy3\cats_and_dogs_small"

train_dir=os.path.join(base_dir,
                       'train')

validation_dir=os.path.join(base_dir,
                            'validation')

test_dir=os.path.join(base_dir,
                      'test')

datagen=ImageDataGenerator(rescale=1./255)

batch_size=20

def extract_features(directory,sample_count):
    
    features=np.zeros(shape=(sample_count,
                             4,
                             4,
                             512))
    
    labels=np.zeros(shape=(sample_count))
    
    generator=datagen.flow_from_directory(directory,
                                          target_size=(150,150),
                                          batch_size=batch_size,
                                          class_mode='binary')
    
    i=0
    
    for inputs_batch,labels_batch in generator:
        
        features_batch=conv_base.predict(inputs_batch)
        
        features[i*batch_size:(i+1)*batch_size]=features_batch
        
        labels[i*batch_size:(i+1)*batch_size]=labels_batch
        
        i+=1
        
        print(i,end='')
        
        if i*batch_size>=sample_count:
            
            break
    
    return features,labels

train_features,train_labels=extract_features(train_dir,
                                             2000)

validation_features,validation_labels=extract_features(validation_dir,
                                                       1000)

test_features,test_labels=extract_features(test_dir,
                                           1000)

#Flat the data

train_features=np.reshape(train_features,
                          (2000,4*4*512))

validation_features=np.reshape(validation_features,
                               (1000,4*4*512))

test_features=np.reshape(test_features,
                         (1000,4*4*512))

#create the dense layers

from keras import models

from keras import layers

from keras import optimizers

model=models.Sequential()

model.add(layers.Dense(256,
                       activation='relu',
                       input_dim=4*4*512))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1,
                       activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history=model.fit(train_features,
                  train_labels,
                  epochs=30,
                  batch_size=20,
                  validation_data=(validation_features,
                                   validation_labels))


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