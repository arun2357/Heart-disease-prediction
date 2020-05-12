#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Predictions using Logistic Regression

#  The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD).The dataset provides the patients’ information. It includes over 4,000 records and 15 attributes.

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


dataset=pd.read_csv(r'C:\Users\Lenovo\Desktop\Arun DS\Task-9\framingham.csv')


# In[6]:


dataset.head()


# In[7]:


dataset.shape


# # EDA

# Lets see how many columns has missing values

# In[8]:


print('Null values in the dataset')
print(dataset.isnull().sum())


# As per the above table we have NAN values in 7 columns

# In[9]:


dataset.drop('glucose',axis=1,inplace=True)


# In[10]:


dataset.head()


# In[11]:


dataset.shape


# In[12]:


print('Null values in the dataset')
print(dataset.isnull().sum())


# In[16]:


dataset.shape


# In[13]:


dataset.dropna(axis=0,inplace=True)


# In[14]:


print(dataset.isnull().sum())


# In[15]:


X=dataset.iloc[:,0:14].values
y=dataset.iloc[:,-1].values


# We have converted all the null values using mean method and now lets split or dataset into training and testing set

# In[16]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


# In[17]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(X_train,y_train)


# In[18]:


y_pred=lr.predict(X_test)


# In[19]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[20]:


from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)


# In[21]:


from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)


# In[ ]:




