#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Dataset loading

# In[4]:


dt = pd.read_csv('Churn_Modelling.csv')
dt.head()


# # Visualizations

# ## Univariate Analysis

# In[5]:


sns.displot(dt.CreditScore)


# In[6]:


sns.displot(dt.Age)


# In[7]:


sns.displot(dt.Tenure)


# ## Bi-Variate Analysis

# In[9]:


sns.lineplot(x=dt.NumOfProducts, y=dt.HasCrCard)


# In[11]:


sns.lineplot(x=dt.Age, y=dt.Tenure)


# In[12]:


sns.lineplot(dt.Age,dt.CreditScore)


# In[14]:


sns.scatterplot(dt.Age,dt.CreditScore)


# In[15]:


sns.lineplot(dt.Tenure,dt.Balance)


# In[17]:


sns.scatterplot(dt.Tenure,dt.Balance)


# In[18]:


sns.lineplot(dt.CreditScore,dt.Balance)


# In[19]:


sns.scatterplot(dt.CreditScore,dt.Balance)


# In[20]:


plt.pie(dt.HasCrCard.value_counts(),[0.2,0],labels=['YES','NO'],autopct="%1.1f%%",colors=['red','blue'])
plt.title('HasCrCard')


# In[21]:


dt.HasCrCard.value_counts()


# In[22]:


sns.barplot(dt.Geography.value_counts().index,dt.Geography.value_counts())


# In[24]:


sns.barplot(dt.Gender.value_counts().index,dt.Gender.value_counts())


# # Multi-Variate Analysis

# In[25]:


dt.hist(figsize=(25,25))


# In[27]:


sns.pairplot(dt)


# In[29]:


plt.pie(dt.Geography.value_counts(),[0,0.1,0.3],shadow=True,labels=['France','Germany','Spain'],autopct="%1.1f%%")
plt.title('Geography')


# # Descriptive statistics on the dataset

# In[30]:


dt.describe()


# In[31]:


dt.Geography.unique()


# In[32]:


dt.Gender.value_counts()


# In[33]:


dt.Geography.value_counts()


# # Handling the missing data and outliers

# In[34]:


sns.boxplot(dt.CreditScore)


# In[38]:


a1=dt.CreditScore.quantile(0.25)  
a3=dt.CreditScore.quantile(0.75)

IQR=a3-a1

upper_limit= a3 + 2.5*IQR
lower_limit= a1 - 2.5*IQR


# In[39]:


print("Upper limit :",upper_limit)
print("Lower limit :",lower_limit)


# In[40]:


dt.median()


# In[41]:


dt['CreditScore']= np.where(dt['CreditScore']<lower_limit,6.520000e+02,dt['CreditScore'])


# In[42]:


sns.boxplot(dt.CreditScore)


# # Label Encoding

# In[43]:


from sklearn.preprocessing import LabelEncoder


# In[44]:


le=LabelEncoder()


# In[45]:


dt.Gender=le.fit_transform(dt.Gender)


# In[47]:


dt.head(10)


# # One hot encoding

# In[49]:


dt_main=pd.get_dummies(dt,columns=['Geography'])
dt_main.head(25)


# In[50]:


dt_main.corr()


# In[52]:


plt.figure(figsize=(10,5))
sns.heatmap(dt_main.corr(),annot=True,cmap="YlGnBu")


# In[53]:


dt_main.corr().Exited.sort_values(ascending=False)


# In[54]:


dt_main.head()


# # Spilting of data for Training and Testing

# # Dependent variable

# In[56]:


y=dt_main['Exited']
print(y)


# # independent variable

# In[58]:


X=dt_main.drop(columns=['Exited'],axis=1)
X.head(5)


# # Scaling

# In[68]:


from sklearn.preprocessing import MinMaxScaler


# In[70]:


scaler=MinMaxScaler()
dt[["RowNumber"]]=scaler.fit_transform(dt[["RowNumber"]])
print(dt)


# # Train Test Split

# In[72]:


from sklearn.model_selection import train_test_split
train_size=0.8


# In[83]:


X=dt.drop(columns=['Tenure']).copy()
y=dt['Tenure']


# In[84]:


X_train,X_rem,y_train,y_rem=train_test_split(X,y,train_size=0.8)
test_size=0.5


# In[85]:


X_valid,X_test,y_valid,y_test=train_test_split(X_rem,y_rem,test_size=0.5)


# In[86]:


print(X_train.shape),print(y_train.shape)
print(X_valid.shape),print(y_valid.shape)
print(X_test.shape),print(y_test.shape)


# In[ ]:




