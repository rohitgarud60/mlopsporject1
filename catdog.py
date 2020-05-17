#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.layers import Convolution2D


# In[3]:


from keras.layers import MaxPooling2D


# In[4]:


from keras.layers import Flatten


# In[5]:


from keras.layers import Dense


# In[6]:


from keras.models import Sequential


# In[7]:


model = Sequential()


# In[8]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))


# In[9]:


model.summary()


# In[10]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[11]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))


# In[12]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:





# In[13]:


model.summary()


# In[14]:


model.add(Flatten())


# In[15]:


model.summary()


# In[16]:


model.add(Dense(units=128, activation='relu'))


# In[17]:


model.add(Dense(units=1, activation='sigmoid'))


# In[18]:


model.summary()


# In[19]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[20]:


from keras_preprocessing.image import ImageDataGenerator


# In[ ]:





# In[39]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('cnn_dataset/training_set/',target_size=(64,64),batch_size=32,class_mode='binary')

test_set = test_datagen.flow_from_directory('cnn_dataset/test_set/', target_size=(64, 64),batch_size=32,class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=100,
        epochs=1,
        validation_data=test_set,
        validation_steps=800)
model.evaluate



# In[ ]:





# In[22]:


model.save('catdogmodel.h5')


# In[23]:


#from keras.models import load_model


# In[25]:


#m = load_model('my.h5')


# In[26]:


#from keras.preprocessing import image


# In[27]:


#test_image = image.load_img('cnn_dataset/single_prediction/cat_or_dog_2.jpg', 
target_size=(64,64)


# In[28]:


#type(test_image)


# In[29]:


#test_image


# In[30]:


#test_image = image.img_to_array(test_image)


# In[31]:


##type(test_image)


# In[32]:


#test_image.shape


# In[33]:


#import numpy as np 


# In[34]:


#test_image = np.expand_dims(test_image, axis=0)


# In[35]:


#test_image.shape


# In[36]:


#result = m.predict(test_image)


# In[37]:


#result


# In[38]:


#if result[0][0] == 1.0:
 #   print('dog')
#else:
#    print('cat')


# In[ ]:





# In[ ]:


#r = training_set.class_indices


# In[ ]:


#r


# In[ ]:





# In[ ]:




