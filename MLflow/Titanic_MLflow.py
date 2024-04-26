#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    try:
        data = pd.read_csv('Titanic+Data+Set.csv')
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )


    # In[9]:


    data.head()


    # ## Exploratory Data Analysis

    # Let's begin some exploratory data analysis! We'll start by checking out missing data!

    # ## Missing Data
    #
    # We can use seaborn to create a simple heatmap to see where we are missing data!

    # In[10]:


    data.isnull()


    # In[12]:


    sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


    # Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"
    #

    # In[13]:


    sns.set_style('whitegrid')
    sns.countplot(x='Survived',data=data)


    # In[14]:


    sns.set_style('whitegrid')
    sns.countplot(x='Survived',hue='Sex',data=data,palette='RdBu_r')


    # In[15]:


    sns.set_style('whitegrid')
    sns.countplot(x='Survived',hue='Pclass',data=data,palette='rainbow')


    # In[16]:


    sns.distplot(data['Age'].dropna(),kde=False,color='darkred',bins=40)


    # In[17]:


    data['Age'].hist(bins=30,color='darkred',alpha=0.3)


    # In[18]:


    sns.countplot(x='SibSp',data=data)


    # In[19]:


    data['Fare'].hist(color='green',bins=40,figsize=(8,4))


    # ___
    # ## Data Cleaning
    # We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation).
    # However we can be smarter about this and check the average age by passenger class. For example:
    #

    # In[20]:


    plt.figure(figsize=(12, 7))
    sns.boxplot(x='Pclass',y='Age',data=data,palette='winter')


    # We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.

    # In[21]:


    def impute_age(cols):
        Age = cols[0]
        Pclass = cols[1]

        if pd.isnull(Age):

            if Pclass == 1:
                return 37

            elif Pclass == 2:
                return 29

            else:
                return 24

        else:
            return Age


    # Now apply that function!

    # In[23]:


    data['Age'] = data[['Age','Pclass']].apply(impute_age,axis=1)


    # Now let's check that heat map again!

    # In[25]:


    sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


    # Let's go ahead and drop the Cabin column and the row in Embarked that is NaN.

    # In[26]:


    data.drop('Cabin',axis=1,inplace=True)


    # In[27]:


    data.head()


    # In[28]:


    data.dropna(inplace=True)


    # ## Converting Categorical Features
    #
    # We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

    # In[29]:


    data.info()


    # In[30]:


    pd.get_dummies(data['Embarked'],drop_first=True).head()


    # In[32]:


    sex = pd.get_dummies(data['Sex'],drop_first=True)
    embark = pd.get_dummies(data['Embarked'],drop_first=True)


    # In[33]:


    data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


    # In[34]:


    data.head()


    # In[36]:


    data = pd.concat([data,sex,embark],axis=1)


    # In[37]:


    data.head()


    # Great! Our data is ready for our model!
    #
    # # Building a Logistic Regression model
    #
    # Let's start by splitting our data into a training set and test set (there is another test.csv file that you can play around with in case you want to use all this data for training).
    #
    # ## Train Test Split

    # In[38]:


    data.drop('Survived',axis=1).head()


    # In[39]:


    data['Survived'].head()


    # In[40]:


    from sklearn.model_selection import train_test_split


    # In[41]:


    X_train, X_test, y_train, y_test = train_test_split(data.drop('Survived',axis=1),
                                                        data['Survived'], test_size=0.30,
                                                        random_state=101)


    # In[42]:


    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)


    # In[43]:


    predictions = logmodel.predict(X_test)


    # In[44]:


    from sklearn.metrics import confusion_matrix


    # In[45]:


    accuracy=confusion_matrix(y_test,predictions)


    # In[46]:


    accuracy


    # In[47]:


    accuracy=accuracy_score(y_test,predictions)
    accuracy


    # In[48]:


    predictions


    # Let's move on to evaluate our model!

    # In[49]:


    from sklearn.metrics import classification_report


    # In[50]:


    print(classification_report(y_test,predictions))


    # ## Log Parameters and Metrics with MLflow:

    # In[55]:


    with mlflow.start_run(nested=True):
        # Log parameters
        mlflow.log_param("penalty", logmodel.penalty)
        mlflow.log_param("C", logmodel.C)

        # Log metrics
        predictions = logmodel.predict(X_test)
        accuracy = accuracy_score(y_test,predictions )
        mlflow.log_metric("accuracy", accuracy)





    mlflow.sklearn.log_model(logmodel, "logistic_regression_model")





    mlflow.sklearn.log_model(logmodel,"logistic_regression_model")






