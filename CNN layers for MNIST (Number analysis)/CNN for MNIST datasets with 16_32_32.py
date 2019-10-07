#######################################
#
#CNN for MNIST 
#
#
#
#
######################################
#
#
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d_1 (Conv2D)            (None, 26, 26, 16)        160       
#_________________________________________________________________
#max_pooling2d_1 (MaxPooling2 (None, 13, 13, 16)        0         
#_________________________________________________________________
#conv2d_2 (Conv2D)            (None, 11, 11, 32)        4640     
#_________________________________________________________________
#max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0         
#_________________________________________________________________
#conv2d_3 (Conv2D)            (None, 3, 3, 64)          9248     
#_________________________________________________________________
#flatten_1 (Flatten)          (None, 576)               0         
#_________________________________________________________________
#dense_1 (Dense)              (None, 64)                18496     
#_________________________________________________________________
#dense_2 (Dense)              (None, 10)                650       
#=================================================================
#Total params: 33,194
#Trainable params: 33,194
#Non-trainable params: 0
#_________________________________________________________________
#      
#      
#        
######################################
from keras import layers

from keras import models

#The sturcture of CNN layers

model=models.Sequential()

#1 Conv2D layer as input layer with 16 neural network union

model.add(layers.Conv2D(16,(3,3),
          activation='relu',
          input_shape=(28,28,1)))

#1 MaxPooling2D layer

model.add(layers.MaxPool2D((2,2)))

#2 Conv2D layer with 32 neural network union

model.add(layers.Conv2D(32,(3,3),
          activation='relu'))

#2 MaxPooling2D layer

model.add(layers.MaxPool2D((2,2)))

#3 Conv2D layer with 32 neural network union

model.add(layers.Conv2D(32,
                        (3,3),
                        activation='relu'))

#the strusture of Conv2D and MaxPooling

#model.summary()

#The last Conv2D output 3D tensor(shape=(3,3,64)).Next, we need to trans it to next NN.

#We flat the 3D tensor to 1D array (3*3*64=576).

model.add(layers.Flatten())

#1D layers with 64 neural network union.

model.add(layers.Dense(64,
                       activation='relu'))

#the output layers(softmax) with 10 NN union.

model.add(layers.Dense(10,
                       activation='softmax'))

#Print the strusture of CNN.

#model.summary()

#After we create the CNN,we take it to train our MNIST datasets with CNN.

from keras.datasets import mnist

from keras.utils import to_categorical

#we input the data from MNIST datasets.

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

#Next,we need to deal with the datas as suitable array (match the CNN as inputing data)

train_images=train_images.reshape((60000,28,28,1))

train_images=train_images.astype('float32')/255

test_images=test_images.reshape((10000,28,28,1))

test_images=test_images.astype('float32')/255

train_labels=to_categorical(train_labels)

test_labels=to_categorical(test_labels)

#We use optimizer as rmsprop and loss fun as categorical_crossentropy.

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


#Train our CNN with MNIST datasets.

history=model.fit(train_images,
                  train_labels,
                  epochs=20,
                  batch_size=64)

#After training our CNN, we use the test_images to test CNN.

test_loss,test_acc=model.evaluate(test_images,
                                  test_labels)

print('test_accuracy')

print(test_acc)

#Print the accuracy and loss quantity of training

import matplotlib.pyplot as plt

acc=history.history['acc']

loss=history.history['loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',
         label='Traning accuracy')

plt.title('The accuracy of training ')

plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',
         label='Training loss quantity')

plt.title('Training loss')

plt.legend()

plt.show() 
