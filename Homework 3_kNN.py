#!/usr/bin/env python
# coding: utf-8

# #    <b> Homework 3 - kNN Neighbors <b>

# <i> Analyst: Marla Gansukh
# 
# Assignment: Homework 2 - Part 1 
# 
# Purpose: Using kNN in predicting tumor type based upon tissue image measurements data. <i>

# >By submitting this code for grading, I confirm the following:
# - that this notebook represents my own unique code created as a solution to the assigned problem 
# - that this notebook was created specifically to satisfy the requirements detailed in this assignment. 
# >other than the textbook and material presented during our class sessions, I DID NOT receive any help in designing and debugging my code from another source.

# In[2]:


import pandas as pd 
from sklearn import neighbors
from sklearn import metrics
from sklearn import model_selection as skms


# In[3]:


tumor = pd.read_csv(r"C:\Users\marla\Downloads\wisc_bc_data.csv", index_col = 'id')
df = tumor 
print(df.info())


# In[4]:


tumor['diagnosis'].value_counts(normalize = True)


# In[5]:


Test_size = 0.2
Random_state = 42
target = tumor['diagnosis']
features = tumor.drop(columns=['diagnosis'])


# In[6]:


tts = skms.train_test_split(features, target,
                           test_size = Test_size, random_state = Random_state)


# In[7]:


(train_ftrs, test_ftrs, train_target, test_target) = tts


# In[8]:


from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
train_std = stdsc.fit_transform(train_ftrs)
test_std = stdsc.transform(test_ftrs)


# In[9]:


K = 3
knn = neighbors.KNeighborsClassifier(n_neighbors = K)
model = knn.fit(train_std, train_target)
preds = model.predict(test_std)


# In[10]:


metrics.accuracy_score(test_target,preds)


# In[11]:


K = 5
knn = neighbors.KNeighborsClassifier(n_neighbors = K)
model = knn.fit(train_std, train_target)
preds = model.predict(test_std)
metrics.accuracy_score(test_target,preds)


# In[12]:


K = 9 
knn = neighbors.KNeighborsClassifier(n_neighbors = K)
model = knn.fit(train_std, train_target)
preds = model.predict(test_std)
metrics.accuracy_score(test_target,preds)


# In[13]:


K = 15 
knn = neighbors.KNeighborsClassifier(n_neighbors = K)
model = knn.fit(train_std, train_target)
preds = model.predict(test_std)
metrics.accuracy_score(test_target,preds)


# <b> According to the results, we have obtained the highest accuracy score of 0.99 at the K=5. Although we have to take other factors such as error rate to determine the ultimate K value for our model, under this condition K=5 is the most suitable K value for utilization. <b>

# <b> The pattern that is visible from this model is that as we choose greater K values, the accuracy score is getting lesser than the previous accuracy score (except the K=3), I believe the K=3 is unsuitable for predicting the pattern since the value is too small to create a model <b>

# <b>  <b>

# # <b> Part - 2 <b>

# In[14]:


titanic =  pd.read_csv(r"C:\Users\marla\Downloads\Titanic_knn.csv")
df = titanic
print(df.info())


# In[15]:


titanic['survived'].value_counts().sort_index()


# In[16]:


Test_size = 0.2
Random_state = 42
target = titanic['survived']
features = titanic.drop(columns=['survived'])


# In[17]:


tts = skms.train_test_split(features, target,
                           test_size = Test_size, random_state = Random_state)


# In[18]:


(train_ftrs, test_ftrs, train_target, test_target) = tts


# In[19]:


from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
train_std = stdsc.fit_transform(train_ftrs)
test_std = stdsc.transform(test_ftrs)


# In[20]:


K = 3
knn = neighbors.KNeighborsClassifier(n_neighbors = K)
model = knn.fit(train_std, train_target)
preds = model.predict(test_std)
metrics.accuracy_score(test_target,preds)


# In[21]:


K = 5
knn = neighbors.KNeighborsClassifier(n_neighbors = K)
model = knn.fit(train_std, train_target)
preds = model.predict(test_std)
metrics.accuracy_score(test_target,preds)


# In[22]:


K = 9 
knn = neighbors.KNeighborsClassifier(n_neighbors = K)
model = knn.fit(train_std, train_target)
preds = model.predict(test_std)
metrics.accuracy_score(test_target,preds)


# In[23]:


K = 15 
knn = neighbors.KNeighborsClassifier(n_neighbors = K)
model = knn.fit(train_std, train_target)
preds = model.predict(test_std)
metrics.accuracy_score(test_target,preds)


# <b> <i> Answer: <i> I believe that the most plausible reason why the accuracy score between two datasets vary so widely is due to the amount of the data we used for each dataset. The second dataset has a larger amount of data compared to the first dataset which helps kNN algorithms to calculate more accurate results without overfitting. <b>

# <b> For the second dataset, a test data is the most accurate at K=5. <b>
