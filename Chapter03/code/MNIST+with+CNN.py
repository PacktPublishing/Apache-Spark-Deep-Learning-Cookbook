
# coding: utf-8

# In[1]:

import tensorflow as tf


# In[2]:

print(tf.__version__)


# In[3]:

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('MNIST/', one_hot=True)


# In[4]:

import os
os.listdir('MNIST/')


# In[5]:

print('Image Inventory')
print('----------')
print('Training: {}'.format(len(data.train.labels)))
print('Testing:  {}'.format(len(data.test.labels)))
print('----------')


# In[6]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[7]:

for i in range(2):
    image = data.train.images[i]
    image = np.array(image, dtype='float')
    label = data.train.labels[i]
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    print('-----------------')
    print(label)
    plt.show()
    


# In[8]:

if not os.path.exists('MNIST/images'):
    os.makedirs('MNIST/images/')
os.chdir('MNIST/images/')


# In[9]:

from matplotlib import image
for i in range(1,10):
    png = data.train.images[i]
    png = np.array(png, dtype='float')
    pixels = png.reshape((28, 28))
    image.imsave('image_no_{}.png'.format(i), pixels, cmap = 'gray')


# In[10]:

print(os.listdir())


# In[11]:

from Augmentor import Pipeline


# In[12]:

augmentor = Pipeline('/home/asherif844/sparkNotebooks/Ch03/MNIST/images')


# In[13]:

augmentor.rotate(probability=0.9, max_left_rotation=25, max_right_rotation=25)


# In[14]:

for i in range(1,3):
    augmentor.sample(10)


# In[15]:

xtrain = data.train.images
ytrain = np.asarray(data.train.labels)
xtest = data.test.images 
ytest = np.asarray(data.test.labels)


# In[16]:

xtrain = xtrain.reshape( xtrain.shape[0],28,28,1)
xtest = xtest.reshape(xtest.shape[0],28,28,1)
ytest= ytest.reshape(ytest.shape[0],10)
ytrain = ytrain.reshape(ytrain.shape[0],10)


# In[17]:

print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)


# In[18]:

import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D

K.set_image_dim_ordering('tf')

model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5),activation='relu', input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='sigmoid'))


# In[19]:

model.compile(optimizer='adam',loss='categorical_crossentropy', 
              metrics=['accuracy'])


# In[20]:

model.fit(xtrain,ytrain,batch_size=512,
          epochs=5,
          validation_data=(xtest, ytest))


# In[21]:

stats = model.evaluate(xtest, ytest)
print('The accuracy rate is {}%'.format(round(stats[1],3)*100))
print('The loss rate is {}%'.format(round(stats[0],2)*100))


# In[22]:

model.summary()


# In[ ]:



