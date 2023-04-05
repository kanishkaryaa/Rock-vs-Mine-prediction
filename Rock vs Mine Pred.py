#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This model is created by kanishkaryaa using the SONAR dataset
get_ipython().system('pip install sklearn')


# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[4]:


#loading the dataset
sonar_data = pd.read_csv('Copy of sonar data.csv',header =None)


# In[5]:


sonar_data.head()


# In[6]:


# number of rows and coulmns
sonar_data.shape


# In[7]:


sonar_data.describe()  #describe functionis retireves statistical inforamtion of the dataset


# In[9]:


sonar_data[60].value_counts()


# In[10]:


sonar_data.groupby(60).mean()


# In[12]:


#seperating data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]


# In[15]:


print(X)
print(Y)


# In[17]:


#training and test data
X_train, X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)


# In[19]:


print(X.shape, X_train.shape, X_test.shape)


# In[22]:


print(X_train)
print(Y_train)


# In[23]:


model = LogisticRegression()


# In[24]:


#training the Logistic Regression model with training data()
model.fit(X_train, Y_train)


# In[25]:


#accuracy on training data
X_train_prediction= model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[26]:


print("Accuracy on training data : ", training_data_accuracy)


# In[29]:


#Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[30]:


print("accuracy on test data : ", test_data_accuracy)


# In[37]:


#Making a predictive system
input_data = (0.0123,0.0309,0.0169,0.0313,0.0358,0.0102,0.0182,0.0579,0.1122,0.0835,0.0548,0.0847,0.2026,0.2557,0.1870,0.2032,0.1463,0.2849,0.5824,0.7728,0.7852,0.8515,0.5312,0.3653,0.5973,0.8275,1.0000,0.8673,0.6301,0.4591,0.3940,0.2576,0.2817,0.2641,0.2757,0.2698,0.3994,0.4576,0.3940,0.2522,0.1782,0.1354,0.0516,0.0337,0.0894,0.0861,0.0872,0.0445,0.0134,0.0217,0.0188,0.0133,0.0265,0.0224,0.0074,0.0118,0.0026,0.0092,0.0009,0.0044)
# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
              
#reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]=='R'):
    print('The object is a Rock')
else:
    print("The object is a mine")


# In[ ]:




