#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns 
import numpy as np 
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataframe=pd.read_csv("abalone.csv") 
dataframe


# # 3.Visualizations

# 3.1 Univariate Analysis

# In[3]:


sns.displot(dataframe.Sex)


# 3.2 Bi-Variate Analysis

# In[4]:


dataframe.plot.line()


# 3.3 Multi-Variate Analysis

# In[5]:


sns.lmplot("Diameter","Length",dataframe,hue="Length",fit_reg=False);


# In[6]:


dataframe.describe()


# 5.Handle the Missing values.

# In[7]:


data1=pd.read_csv("abalone.csv")
pd.isnull(data1["Sex"])


# # 6.Find the outliers and replace the outliers
# 

# In[9]:


dataframe["Rings"]=np.where(dataframe["Rings"]>10,np.median,dataframe["Rings"]) 
dataframe["Rings"]


# # 7.Check for Categorical columns and perform encoding

# In[10]:


pd.get_dummies(dataframe,columns=["Sex","Length"],prefix=["Length","Sex"]).head()


# 8.Split the data into dependent and independent variables 

# # 8.1 Split the data into Independent variables.

# In[11]:


X=dataframe.iloc[:,:-2].values 
print(X)


# 8.2 Split the data into Dependent variables

# In[13]:


Y=dataframe.iloc[:,-1].values 
print(Y)


# 9.Scale the independen tvariables

# In[14]:


import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
scaler=MinMaxScaler() 


# In[16]:


dataframe[["Length"]]=scaler.fit_transform(dataframe[["Length"]]) 
print(dataframe)


# 10.Split the data into training and testing

# In[17]:


from sklearn.model_selection import train_test_split
train_size_A=0.8
X=dataframe.drop(columns=['Sex']).copy()
y=dataframe['Sex']
X_train,X_rem,y_train,y_rem=train_test_split(X,y,train_size=0.8)
test_size=0.5
X_valid,X_test,y_valid,y_test=train_test_split(X_rem,y_rem,test_size=0.5)
print(X_train.shape),print(y_train.shape)
print(X_valid.shape),print(y_valid.shape)
print(X_test.shape),print(y_test.shape)


# # 11.Build the Model

# In[19]:


test_size=0.33 
seed_1=7 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=seed_1)


# 12.Train the model

# In[20]:


X_train


# In[21]:


y_train


# 13.Test the model

# In[22]:


X_test


# In[23]:


y_test


# 14.Measure the performance using Metrics

# In[24]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error
X_train=[5,-1,2,10]
y_test=[3.5,-0.9,2,9.9] 
print('RSquared=',r2_score(X_train,y_test)) 
print('MAE=',mean_absolute_error(X_train,y_test)) 
print('MSE=',mean_squared_error(X_train,y_test))


# In[ ]:




