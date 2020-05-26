#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset = pd.read_csv('SalaryData.csv')


# In[3]:


dataset.head()


# In[5]:


y=dataset['Salary']


# In[9]:


x=dataset['YearsExperience']


# In[10]:


x


# In[11]:


type(x)


# In[12]:


Z=x.values.reshape(30,1)


# In[13]:


Z


# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


model=LinearRegression()


# In[18]:


model.fit(Z,y)


# In[19]:


model.predict([[1.7]])


# In[ ]:




