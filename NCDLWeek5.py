#!/usr/bin/env python
# coding: utf-8

# **Task 1:**
# ### Problem - Develop and evaluate an image classifier using Convolution Neural Network.
# 
# ### The Data = CIFAR 10 datset.
# 
# CIFAR-10 is a dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.
# https://www.cs.toronto.edu/~kriz/cifar.html
# 
# 
# # CIFAR-10 Multiple Classes
# # New section
# Example of using Keras for CNNs. Use a famous data set, the CIFAR-10 dataset which consists of 10 different image types.

# In[1]:


# import libraries

import pandas as pd
import numpy as np


# Following code loads the CIFAR 10 datset.

# In[2]:


from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[3]:


# print shapes of training and test data. Check how the data looks


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[4]:


# print the shape of one image

x_train[0].shape


# ### RGB Colour Model)

# ![image-2.png](attachment:image-2.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[ ]:





# In[5]:


# show any sample image in the dataset. Use - plt.imshow()

import matplotlib.pyplot as plt


# HORSE
plt.imshow(x_train[7]);


# In[6]:


# Show any other image

# TRUCK
plt.imshow(x_train[16]);


# In[ ]:





# In[7]:


# Class names in the CIFAR-10 dataset
classes=['aircraft', 'car', 'bird', 'cat', 'deer',
'dog', 'frog', 'hors', 'ship', 'truck']


# In[8]:


# Random Demo example from database

import random # random number generator

from PIL import Image


n = random.randint(0, x_test.shape[0])

# Image.fromarray - for drawing an image from CIFAR-10
plt.imshow(Image.fromarray(x_train[n]))
plt.show()

print("Picture number in the DataBase:", n)
print("Class number:", y_train[n])
print("Class name:", classes[y_train[n][0]])


# You can run this cell many times, and each timeyou will get a new picture


# In[ ]:





# # PreProcessing

# In[9]:


# print one image. Check how pixels look like. 

print(x_train[0])


# In[10]:


# print shape of any image i.e particular value in the x_train

x_train[0].shape


# In[11]:


# print the maximum value in x_train. Hint - Use .max()

x_train.max()


# In[12]:


x_train = x_train/225
x_test  = x_test/255


# In[13]:


# Print the shape of x_train again.

x_train.shape


# In[14]:


# Print the shape of y_train again

y_train.shape


# # One hot vector encoding.
# ![vector.png](data:image/png;base64,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)
# 
# 
# The "to_categorical()" is converting the integer value to binary categorical matrix :: https://keras.io/api/utils/python_utils/#to_categorical-functionLinks to an external site.

# In[15]:


from tensorflow.keras.utils import to_categorical


# In[16]:


# print shape of y_train

print(y_train.shape)


# Print shape of y_train [0]

print(y_train[0].shape)


# In[17]:


# code to use to_categorical to convert integers to numbers. 
# Assign the new array to the variable y_cat_train

y_cat_train = to_categorical(y_train,10)


# In[18]:


# Print shape of the array y_cat_train

y_cat_train.shape


# In[19]:


print(y_train[0])

# print any single value in the array y_cat_train

print(y_cat_train[0])


# In[20]:


# Convert y_test to the encoded vector in same manner/ Assign it to the variable y_cat_test

y_cat_test = to_categorical(y_test,10)


# In[21]:


# Print the shapes to look how y_cat_test looks.

print(y_cat_test.shape)

print(y_test[0])
print(y_cat_test[0])


# ----------
# # Building the Model

# In[22]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten


# In[23]:


model = Sequential()


# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))


# additional convolution and pooling layers with some choice of filters, strides, and activation function

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=64, kernel_size=(4,4), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())


# Add few dense layers. 

# 256 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(256, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))




# In[24]:


model.summary()


# ### We could use early stopping in Keras to break from the training.
# ### What this does is for successive iterations it monitors the loss. If the loss does not decreases for a certain number of iterations denoted by variable 'patience' then the training stops.
# ### Following code shows how you can use early stopping.
# 

# In[25]:


from tensorflow.keras.callbacks import EarlyStopping


# In[26]:


early_stop = EarlyStopping(monitor='val_loss',patience=3)


# In[27]:


#EarlyStopping?


# In[28]:


# compile the models (model.compile). Use 'categorical cross entropy as the loss function'

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[29]:


# fit the model (model.fit). Use additional parameter -  callbacks = [early_stop] to eanable early stopping. 

history = model.fit( x_train, y_cat_train,
                # the number of times of training using the whole dataset
                    epochs=15,
                    
                    validation_data =(x_test,y_cat_test),
                    
                # Boolean (whether to shuffle the training data before each epoch)
                    shuffle = True,
                
                # 1 = progress bar printing
                    verbose = 1,
                    
                # early stopping    
                    callbacks=[early_stop] )



# In[ ]:





# ### We can save the model in a file. Following code shows how to do this.

# In[30]:


# Careful, don't overwrite file!

model.save('cifar_10epochs.h5')


# ### We can save the loss at every step. Following code shows how to do this. 

# In[31]:


losses = pd.DataFrame(model.history.history)


# In[32]:


# Print first few rows of losses.

losses.head(8)


# In[ ]:





# ### Visualize a plot between accuracy vs val_accuracy for various steps.
# 
# ### x axis will have the steps (epochs) 
# ### y axis will have accuracy and val_accuracy and 
# 
# 
# 
# 
# 
# 

# In[33]:


# plot accuracy and val_accuracy wrt the epochs

losses[['accuracy','val_accuracy']].plot()


# In[34]:


# printing Accuracy for the neural network training process

history_dict = history.history
plt.style.use('seaborn-darkgrid')

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(acc_values) + 1)

plt.figure(num=1, figsize=(15,7))
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[35]:


# plot loss and val_loss wrt the epochs

losses[['loss','val_loss']].plot()


# In[36]:


# printing Loss function (categorical_crossentropy) for the neural network training process

history_dict = history.history
plt.style.use('seaborn-darkgrid')

acc_values = history_dict['loss']
val_acc_values = history_dict['val_loss']
epochs = range(1, len(acc_values) + 1)

plt.figure(num=1, figsize=(15,7))
plt.plot(epochs, acc_values, 'bo', label='Training Loss (categorical_crossentropy)')
plt.plot(epochs, val_acc_values, 'b', label='Validation Loss (categorical_crossentropy)')
plt.xlabel('Epochs')
plt.ylabel('categorical_crossentropy')
plt.legend()

plt.show()


# In[ ]:





# ### You could print the metrics name that model have used.
# 
# ### Following code shows how to do this.

# In[37]:


model.metrics_names


# ### Following code shows how to evaluate your model.

# In[38]:


model.evaluate(x_test,y_cat_test,verbose=1)


# ### Make a prediction on test data set.   

# In[39]:


# make predictions on the test dataset. 
# Store the prediction in variable 'predicrion'. Hint - Use model.predict()

#prediction = model.predict(x_test)
prediction = np.argmax(model.predict(x_test), axis = -1)


# In[40]:


prediction


# In[41]:


prediction.shape


# In[42]:


#np.argmax?


# In[ ]:





# ### We could get the full classification report. Following code does that.

# In[43]:


from sklearn.metrics import classification_report

print(classification_report(y_test, prediction))


# In[ ]:





# ### We could also get and plot confusion matrix. Follwoing code does that. 

# In[44]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,prediction)


# In[45]:


import seaborn as sns

plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,prediction),annot=True)

# https://github.com/matplotlib/matplotlib/issues/14751


# # Following code makes a prediction on specific image.

# In[46]:


my_image = x_test[16]


# In[47]:


plt.imshow(my_image);


# In[48]:


# SHAPE --> (num_images,width,height,color_channels)

#model.predict_classes(my_image.reshape(1,32,32,3))

np.argmax(model.predict(my_image.reshape(1,32,32,3)), axis = -1)


# In[49]:


#


# In[ ]:





# In[56]:


# image recognition demonstration
# neural network operation

from tensorflow.keras.preprocessing.image import array_to_img

# we take a random element 'random.randint()' from the test sample
# and observe: the neural network will guess or not
index = random.randint(0, x_test.shape[0])
plt.imshow(array_to_img(x_test[index]))

# test image conversion
x = x_test[index]
x = np.expand_dims(x, axis=0)

# start recognition
prediction = model.predict(x)
sample = x

# converting the result from one hot encoding format
ans = np.argmax(prediction)

fig = plt.figure(figsize=(12,4))

ax = fig.add_subplot(1, 2, 2)
bar_list = ax.bar(np.arange(10), prediction[0], align='center')
bar_list[ans].set_color('g')
ax.set_xticks(np.arange(10))
ax.set_xlim([-1, 10])
ax.grid('on')

plt.show()

print('The predicted answer: {}'.format((classes[ans])), "\n",
     'Correct answer: {}'.format(classes[y_test[index][0]]) )

print(classes)


# You can run this cell many times, and each time you will get a new picture,which
# this neural network will recognise with the probability indicated on the graph.


# In[ ]:





# **Task 2:**
# 
# Evaluate your model for different optimizers available in the Keras. 
# 
# - Store the optimizers in an array.
# - Use for loop to fit, compile, and  test your model.
# - Plot the accuracy vs optimizer
# 

# In[51]:


# 


# In[52]:


optimizers = ['SGD', 'RMSprop', 'Adam', 'Adadelta']
scores = []


# In[57]:


for row in optimizers:
  model.compile(loss='categorical_crossentropy', optimizer= row, metrics=['accuracy'])
  model.fit(x_train,y_cat_train,epochs=3,validation_data=(x_test,y_cat_test),callbacks=[early_stop])
  scores.append(model.evaluate(x_test,y_cat_test,verbose=0))


# In[58]:


plt.plot (optimizers, scores);


# In[59]:


print(scores)


# In[ ]:





# In[ ]:





# In[ ]:




