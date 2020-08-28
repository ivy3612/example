#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
import pandas as pd
np.random.seed(10)


# In[2]:


train = []
for i in range(0,2528):
    train.append(cv2.imread("D:/AOI/train_images/train_" + "%05d" % i + ".png"))

train = np.array(train) 


# In[3]:


train.shape


# In[4]:


Images2 = []
for i in range(0,2528):
    image = cv2.resize(train[i],(32,32))
    Images2.append(image)
Images3 = np.array(Images2) 


# In[5]:


Images3.shape


# In[6]:


test = []
for i in range(0,5072):
    test.append(cv2.imread("D:/AOI/test_images/test_" + "%05d" % i + ".png"))

test = np.array(test) 


# In[7]:


test.shape


# In[8]:


test2 = []
for i in range(5072,10142):
    test2.append(cv2.imread("D:/AOI/test_images/test_" + "%05d" % i + ".png"))

test2 = np.array(test2) 


# In[9]:


test2.shape


# In[10]:


Img2 = []
for i in range(0,5072):
    Img = cv2.resize(test[i],(32,32))
    Img2.append(Img)
Img3 = np.array(Img2) 


# In[11]:


Imge2 = []
for i in range(0,5070):
    Imge = cv2.resize(test2[i],(32,32))
    Imge2.append(Imge)
Imge3 = np.array(Imge2) 


# In[12]:


Images = np.concatenate([Img3,Imge3])


# In[13]:


Images3_normalize = Images3.astype('float32') / 255.0
Images_normalize = Images.astype('float32') / 255.0


# In[14]:


train_labels=pd.read_csv('D:/AOI/train.csv')
len(train_labels)


# In[15]:


Labels = np.array(train_labels)


# In[16]:


labels=Labels[:,1]


# In[17]:


from keras.utils import np_utils
labelsOneHot = np_utils.to_categorical(labels)


# In[18]:


labelsOneHot[:3]


# In[71]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

import matplotlib.pyplot as plt


# In[113]:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Dropout(0.1))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D((2, 2)))


# In[114]:


model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(6, activation='softmax'))


# In[115]:


model.summary()


# In[116]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[126]:


trained = model.fit(Images3_normalize,labelsOneHot,validation_split=0.25,epochs=20,batch_size=12, verbose=1)


# In[127]:


plt.plot(trained.history['accuracy'])
plt.plot(trained.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(trained.history['loss'])
plt.plot(trained.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[128]:


scores = model.evaluate(Images3_normalize, 
                        labelsOneHot, verbose=0)
scores[1]


# In[129]:


prediction=model.predict_classes(Images_normalize)


# In[130]:


prediction[:3]


# In[131]:


pd_data = pd.DataFrame(prediction,columns=['label'])
pd_data.to_csv('D:/AOI/data1.csv')


# In[ ]:




