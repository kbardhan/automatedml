#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications import resnet50


# In[3]:


# pre trained weights : create my model
model = resnet50.ResNet50(weights='imagenet')


# In[6]:


model.layers[0].input


# In[7]:


from keras.preprocessing import image


# In[8]:


img = image.load_img('cat_or_dog_1.jpg', target_size=(224, 224))


# In[9]:


img


# In[10]:


img.size


# In[11]:


type(img)


# In[12]:


img_np = image.img_to_array(img)


# In[13]:


img_np.shape


# In[14]:


type(img_np)


# In[15]:


import numpy as np


# In[16]:


a = np.array([1,2])


# In[17]:


a.shape


# In[18]:


b = np.expand_dims(a, axis=0)


# In[19]:


b.shape


# In[20]:


b[0]


# In[21]:


ae = np.expand_dims(img_np, axis=0)


# In[22]:


ae.shape


# In[23]:


from keras.applications.resnet50 import decode_predictions


# In[24]:


from keras.applications.resnet50 import preprocess_input


# In[25]:


finalimg = preprocess_input(ae)


# In[ ]:





# In[26]:


pred = model.predict(finalimg)


# In[27]:


decode_predictions(pred, top=3)[0]


# In[ ]:




