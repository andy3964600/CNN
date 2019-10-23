# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:29:47 2019

@author: andy3
"""

#################################################################################################
#
#
#Use the CNN to solve the sequence data
#
#
#
#################################################################################################
#
####Purpose:We want to slove the sequence data by 1-D CNN , the ddatasets we can take it from IMDB
#
####The setup can be followed by:
#
####1. Prepare and deal with the data to staisfy the shape of 1-D CNN layers.
#
####2. Create the triple 1-D CNN to implement the mission.
#
#################################################################################################


#First step: Prepare the data of IMDB

from keras.datasets import imdb

from keras.preprocessing import sequence

max_features=10000

max_len=500

print('Stretching the data... please wait')

import numpy as np
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# restore np.load for future normal usage
np.load = np_load_old

#25000 training sequences data
print(len(x_train),'training sequences data')

#25000 testing sequences data
print(len(x_test),'testing sequences data')

x_train=sequence.pad_sequences(x_train,maxlen=max_len)

x_test=sequence.pad_sequences(x_test,maxlen=max_len)

#training sequences tensor of shape=(25000,500)
print('x_train shape:',x_train.shape)

#testing sequences tensor of shape=(25000,500)
print('x_test shape:',x_test.shape)

def generator(data, lookback, delay, min_index, max_index,
                shuffle=False, batch_size=128, step=6):
    
    if max_index is None:
        
        max_index = len(data) - delay - 1
    
    i = min_index + lookback
    
    while 1:
        
        if shuffle:
            
            rows = np.random.randint(
                    
                    min_index + lookback, max_index, size=batch_size)
        else:
            
            if i + batch_size >= max_index:
                
                i = min_index + lookback
            
            rows = np.arange(i, min(i + batch_size, max_index))
            
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        
        targets = np.zeros((len(rows),))
        
        for j, row in enumerate(rows):
            
            indices = range(rows[j] - lookback, rows[j], step)
            
            samples[j] = data[indices]
            
            targets[j] = data[rows[j] + delay][1]
        
        yield samples, targets    
        
#After we created the generator function,  we also need to create 3 generators:training sets,validation sets,testing sets.

#Create the training sets,validation sets,testing sets

lookback=1440

step=6

delay=144

batch_size=128

#Training sets generator:

train_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=0,
                        max_index=200000,
                        shuffle=True,
                        step=step,
                        batch_size=batch_size)

#validation sets generator:

val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=200001,
                        max_index=300000,
                        step=step,
                        batch_size=batch_size)

#testing sets generator:

test_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=300001,
                        max_index=None,
                        step=step,
                        batch_size=batch_size)
#the number of validating rounds:

val_steps = (300000 - 200001 - lookback)  // batch_size # How many steps to draw from
            # val_gen in order to see the entire validation set

#the number of testing rounds:

test_steps = (len(float_data) - 300001 - lookback)  // batch_size # How many steps to draw

#Second step: Create the 2 1-D CNN to implement the mission

from keras.models import Sequential

from keras import layers

from keras.optimizers import RMSprop

#Now, start to create the CNN
Neural_network=Sequential()

#Use the Embedding layers as our inputing layers
Neural_network.add(layers.Embedding(max_features,
                                    128,
                                    input_length=max_len))

#Use the conv1d layers as our forst hidden layers
Neural_network.add(layers.Conv1D(32,
                                 7,
                                 activation='relu'))

#Use the Maxpooling layers as our second hidden layers
Neural_network.add(layers.MaxPool1D(5))

#Use the Conv1d layers as our third
Neural_network.add(layers.Conv1D(32,
                                 7,
                                 activation='relu'))

#Use the GlobalMaxPooling layers
Neural_network.add(layers.GlobalAveragePooling1D())

#Use the Dense layers as our output layses
Neural_network.add(layers.Dense(1))

#The strusture of our total layers:

Neural_network.summary()

#Final,we need the optimizer,loss,metrics
Neural_network.compile(optimizer=RMSprop(lr=1e-4),
                       loss='binary_crossentropy',
                       metrics=['acc'])

history=Neural_network.fit(x_train,y_train,
                           epochs=10,
                           batch_size=128,
                           validation_split=0.2)

#Conpare the relation of training and validation


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




