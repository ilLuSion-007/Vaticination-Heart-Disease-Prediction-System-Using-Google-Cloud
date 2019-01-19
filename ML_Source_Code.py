
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
columns = ["age", "sex", "cpp", "restbp", "chol", "fbs", "restecg","thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
df= pd.read_csv("https://raw.githubusercontent.com/mayankjoshi12/Syncronization-Oveer-FTP/master/processed_cleveland_data1.csv", header=None,names=columns)
df.head()


# In[2]:


y=df['num']


# In[3]:


X=df[ ["age", "sex", "cpp", "restbp", "chol", "fbs", "restecg","thalach", "exang", "oldpeak", "slope", "ca", "thal"]]


# In[4]:


from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response values for the observations in X
logreg.predict(X)


# In[5]:


y_pred = logreg.predict(X)
# compute classification accuracy for the logistic regression model 
#Training Accuracy for logistic Regression
from sklearn import metrics
print(metrics.accuracy_score(y, y_pred))


# In[6]:


from sklearn.neighbors import KNeighborsClassifier
# try K=1 through K=25 and record training accuracy
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y)
    y_pred = knn.predict(X)
    scores.append(metrics.accuracy_score(y, y_pred))
    # import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Training Accuracy')
    


# In[7]:


# Select k=1, for training Accuracy
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))


# In[8]:


#Training Accuracy on Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()
nb.fit(X,y)
y_pred = nb.predict(X)
print(metrics.accuracy_score(y, y_pred))


# In[9]:


#Training Accuracy on Decision Trees
from sklearn import tree
dt=tree.DecisionTreeClassifier()
dt.fit(X,y)
y_pred = dt.predict(X)
print(metrics.accuracy_score(y, y_pred))


# In[10]:


#Training Accuracy on SVM 
from sklearn.svm import SVC
svm=SVC()
svm.fit(X,y)
y_pred = svm.predict(X)
print(metrics.accuracy_score(y, y_pred))


# In[11]:


#Now we will check Testing Accuracy as training accuracy overfits the data 
# STEP 1: split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

# STEP 2: train the model on the training set
# First we will use Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# STEP 3: make predictions on the testing set
y_pred = logreg.predict(X_test)

# compare actual response values (y_test) with predicted response values (y_pred)
print(metrics.accuracy_score(y_test, y_pred))


# In[12]:


from sklearn.neighbors import KNeighborsClassifier
# try K=1 through K=30 and record training accuracy
k_range = list(range(1, 31))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
    # import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
    


# In[13]:


# Select k=15, for testing Accuracy using the plot above
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[14]:


#Testing Accuracy on Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()
nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[15]:


#Testing Accuracy on Decision Trees
from sklearn import tree
dt=tree.DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[16]:


#Testing Accuracy on SVM 
from sklearn.svm import SVC
svm=SVC()
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[17]:


# k-fold cross-validation to improve testing accuracy   

from sklearn.model_selection import cross_val_score


cross_range = list(range(2, 31))
scores = []
for c in cross_range:
    scores.append(cross_val_score(logreg, X_train, y_train, cv=c, scoring='accuracy').mean())



import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
plt.plot(cross_range, scores)
plt.xlabel('Value of cv for Cross Validation in Logistic Regression')
plt.ylabel('Testing Accuracy')
    


# In[18]:


#Best value of k-Fold cross validation is at cv=29 for logistic Regression
cross_val_score(logreg, X_train, y_train, cv=29, scoring='accuracy').mean()


# In[19]:


cross_range = list(range(2, 31))
scores = []
for c in cross_range:
    scores.append(cross_val_score(knn, X_train, y_train, cv=c, scoring='accuracy').mean())
import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
plt.plot(cross_range, scores)
plt.xlabel('Value of cv for Cross Validation in KNN')
plt.ylabel('Testing Accuracy')
        


# In[20]:


#Best value of k-Fold cross validation is at cv=22 for KNN
cross_val_score(knn, X_train, y_train, cv=22, scoring='accuracy').mean()


# In[21]:


cross_range = list(range(2, 31))
scores = []
for c in cross_range:
    scores.append(cross_val_score(nb, X_train, y_train, cv=c, scoring='accuracy').mean())
import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
plt.plot(cross_range, scores)
plt.xlabel('Value of cv for Cross Validation in Naive Bayes')
plt.ylabel('Testing Accuracy')
        


# In[22]:


#Best value of k-Fold cross validation is at cv=20 for Naive Bayes
cross_val_score(nb, X_train, y_train, cv=20, scoring='accuracy').mean()


# In[23]:


cross_range = list(range(2, 31))
scores = []
for c in cross_range:
    scores.append(cross_val_score(dt, X_train, y_train, cv=c, scoring='accuracy').mean())
import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
plt.plot(cross_range, scores)
plt.xlabel('Value of cv for Cross Validation in Decision Trees')
plt.ylabel('Testing Accuracy')
        


# In[24]:


#Best value of k-Fold cross validation is at cv=15 for Decision Trees
cross_val_score(dt, X_train, y_train, cv=15, scoring='accuracy').mean()


# In[25]:


cross_range = list(range(2, 31))
scores = []
for c in cross_range:
    scores.append(cross_val_score(svm, X_train, y_train, cv=c, scoring='accuracy').mean())
import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
plt.plot(cross_range, scores)
plt.xlabel('Value of cv for Cross Validation in SVM')
plt.ylabel('Testing Accuracy')
        


# In[26]:


#High value of cv in cross validation will lead to high variance and thus Overfitting problem
#Best value of k-Fold cross validation is at cv=118 for SVM
cross_val_score(svm, X_train, y_train, cv=20, scoring='accuracy').mean()


# In[44]:


#Finally we are selecting Naive Bayes as it is having higher accuracy after taking into consideration the problem of Over-fitting
from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()
nb.fit(X_train,y_train)
#Taking user define input to test the model
X_new= pd.read_csv("http://vaticination.ga/dynamic_data.php")
y_pred = nb.predict(X_new)
print(' '.join(map(str,y_pred)))


# In[49]:


#Feature selection start.Till now accuracy is 0.522
#age
feature_cols = [ "sex", "cpp", "restbp", "chol", "fbs", "restecg","thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# use the list to select a subset of the original DataFrame
X1 = df[feature_cols]

# select a Series from the DataFrame
y1 = df.num

# split into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=1)

# fit the model to the training data (learn the coefficients)
nb.fit(X_train1, y_train1)

# make predictions on the testing set
y_pred1 = nb.predict(X_test1)
print(metrics.accuracy_score(y_test1, y_pred1))


# In[53]:


#sex
feature_cols = ["age", "cpp", "restbp", "chol", "fbs", "restecg","thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# use the list to select a subset of the original DataFrame
X1 = df[feature_cols]

# select a Series from the DataFrame
y1 = df.num

# split into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=1)

# fit the model to the training data (learn the coefficients)
nb.fit(X_train1, y_train1)

# make predictions on the testing set
y_pred1 = nb.predict(X_test1)
print(metrics.accuracy_score(y_test1, y_pred1))


# In[54]:


#cpp
feature_cols = ["age", "sex", "restbp", "chol", "fbs", "restecg","thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# use the list to select a subset of the original DataFrame
X1 = df[feature_cols]

# select a Series from the DataFrame
y1 = df.num

# split into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=1)

# fit the model to the training data (learn the coefficients)
nb.fit(X_train1, y_train1)

# make predictions on the testing set
y_pred1 = nb.predict(X_test1)
print(metrics.accuracy_score(y_test1, y_pred1))


# In[55]:


#restbp
feature_cols = ["age", "sex","cpp", "chol", "fbs", "restecg","thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# use the list to select a subset of the original DataFrame
X1 = df[feature_cols]

# select a Series from the DataFrame
y1 = df.num

# split into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=1)

# fit the model to the training data (learn the coefficients)
nb.fit(X_train1, y_train1)

# make predictions on the testing set
y_pred1 = nb.predict(X_test1)
print(metrics.accuracy_score(y_test1, y_pred1))


# In[56]:


#chol
feature_cols = ["age", "sex","cpp", "restbp", "fbs", "restecg","thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# use the list to select a subset of the original DataFrame
X1 = df[feature_cols]

# select a Series from the DataFrame
y1 = df.num

# split into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=1)

# fit the model to the training data (learn the coefficients)
nb.fit(X_train1, y_train1)

# make predictions on the testing set
y_pred1 = nb.predict(X_test1)
print(metrics.accuracy_score(y_test1, y_pred1))


# In[57]:


#fbs
feature_cols = ["age", "sex","cpp", "restbp","chol", "restecg","thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# use the list to select a subset of the original DataFrame
X1 = df[feature_cols]

# select a Series from the DataFrame
y1 = df.num

# split into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=1)

# fit the model to the training data (learn the coefficients)
nb.fit(X_train1, y_train1)

# make predictions on the testing set
y_pred1 = nb.predict(X_test1)
print(metrics.accuracy_score(y_test1, y_pred1))


# In[59]:


#restecg
feature_cols = ["age", "sex","cpp", "restbp","chol","fbs", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# use the list to select a subset of the original DataFrame
X1 = df[feature_cols]

# select a Series from the DataFrame
y1 = df.num

# split into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=1)

# fit the model to the training data (learn the coefficients)
nb.fit(X_train1, y_train1)

# make predictions on the testing set
y_pred1 = nb.predict(X_test1)
print(metrics.accuracy_score(y_test1, y_pred1))


# In[60]:


#thalach
feature_cols = ["age", "sex","cpp", "restbp","chol","fbs", "restecg", "exang", "oldpeak", "slope", "ca", "thal"]

# use the list to select a subset of the original DataFrame
X1 = df[feature_cols]

# select a Series from the DataFrame
y1 = df.num

# split into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=1)

# fit the model to the training data (learn the coefficients)
nb.fit(X_train1, y_train1)

# make predictions on the testing set
y_pred1 = nb.predict(X_test1)
print(metrics.accuracy_score(y_test1, y_pred1))


# In[61]:


#exang
feature_cols = ["age", "sex","cpp", "restbp","chol","fbs", "restecg","thalach", "oldpeak", "slope", "ca", "thal"]

# use the list to select a subset of the original DataFrame
X1 = df[feature_cols]

# select a Series from the DataFrame
y1 = df.num

# split into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=1)

# fit the model to the training data (learn the coefficients)
nb.fit(X_train1, y_train1)

# make predictions on the testing set
y_pred1 = nb.predict(X_test1)
print(metrics.accuracy_score(y_test1, y_pred1))


# In[62]:


#oldpeak
feature_cols = ["age", "sex","cpp", "restbp","chol","fbs", "restecg","thalach","exang", "slope", "ca", "thal"]

# use the list to select a subset of the original DataFrame
X1 = df[feature_cols]

# select a Series from the DataFrame
y1 = df.num

# split into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=1)

# fit the model to the training data (learn the coefficients)
nb.fit(X_train1, y_train1)

# make predictions on the testing set
y_pred1 = nb.predict(X_test1)
print(metrics.accuracy_score(y_test1, y_pred1))


# In[63]:


#slope
feature_cols = ["age", "sex","cpp", "restbp","chol","fbs", "restecg","thalach","exang","oldpeak", "ca", "thal"]

# use the list to select a subset of the original DataFrame
X1 = df[feature_cols]

# select a Series from the DataFrame
y1 = df.num

# split into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=1)

# fit the model to the training data (learn the coefficients)
nb.fit(X_train1, y_train1)

# make predictions on the testing set
y_pred1 = nb.predict(X_test1)
print(metrics.accuracy_score(y_test1, y_pred1))


# In[64]:


#ca
feature_cols = ["age", "sex","cpp", "restbp","chol","fbs", "restecg","thalach","exang","oldpeak","slope", "thal"]

# use the list to select a subset of the original DataFrame
X1 = df[feature_cols]

# select a Series from the DataFrame
y1 = df.num

# split into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=1)

# fit the model to the training data (learn the coefficients)
nb.fit(X_train1, y_train1)

# make predictions on the testing set
y_pred1 = nb.predict(X_test1)
print(metrics.accuracy_score(y_test1, y_pred1))


# In[65]:


#thal
feature_cols = ["age", "sex","cpp", "restbp","chol","fbs", "restecg","thalach","exang","oldpeak","slope", "ca"]

# use the list to select a subset of the original DataFrame
X1 = df[feature_cols]

# select a Series from the DataFrame
y1 = df.num

# split into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=1)

# fit the model to the training data (learn the coefficients)
nb.fit(X_train1, y_train1)

# make predictions on the testing set
y_pred1 = nb.predict(X_test1)
print(metrics.accuracy_score(y_test1, y_pred1))


# In[71]:


#sex,thalach:-
feature_cols = ["age","cpp","restbp","chol","fbs", "restecg","exang","oldpeak","slope", "ca","thal"]

# use the list to select a subset of the original DataFrame
X1 = df[feature_cols]

# select a Series from the DataFrame
y1 = df.num

# split into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=1)

# fit the model to the training data (learn the coefficients)
nb.fit(X_train1, y_train1)

# make predictions on the testing set
y_pred1 = nb.predict(X_test1)
print(metrics.accuracy_score(y_test1, y_pred1))

