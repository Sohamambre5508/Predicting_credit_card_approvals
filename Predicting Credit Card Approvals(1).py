#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm


# In[2]:


df = pd.read_csv('loan.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df['loanAmount_log']=np.log(df['LoanAmount'])
df['loanAmount_log'].hist(bins=30)


# In[7]:


df.isnull().sum()


# In[8]:


df['TotalIncome']= df['ApplicantIncome']+df["CoapplicantIncome"]
df['TotalIncome_log']=np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)


# In[9]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)


df.LoanAmount=df.LoanAmount.fillna(df.LoanAmount.mean())
df.loanAmount_log=df.loanAmount_log.fillna(df.loanAmount_log.mean())


df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)



# In[10]:


df.isnull().sum()


# In[11]:


x=df.iloc[:,np.r_[1:5,9:11,13:15]].values
y=df.iloc[:,12].values


# In[12]:


x


# In[13]:


y


# In[14]:


print('per of missing gender is %2f%%' %((df['Gender'].isnull().sum()/df.shape[0])*100))


# In[15]:


print('number of people who take loan as group by gender:')
print(df['Gender'].value_counts())
sns.countplot(x='Gender', data=df,palette='Set1')


# In[16]:


print('number of people who take loan as group by marital status:')
print(df['Married'].value_counts())
sns.countplot(x='Married', data=df,palette='Set1')


# In[17]:


print('number pf people who take loan as group by dependents:')
print(df['Dependents'].value_counts())
sns.countplot(x='Dependents', data=df,palette='Set1')


# In[18]:


print('number of people who take loan as group by self_employed:')
print(df['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed', data=df,palette='Set1')


# In[19]:


print('number of people who take loan as group by Loanamount:')
print(df['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount', data=df,palette='Set1')


# In[20]:


print('number of people who take loan as group by credit history:')
print(df['Credit_History'].value_counts())
sns.countplot(x='Credit_History', data=df,palette='Set1')


# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[22]:


print(x_train)


# In[23]:


from sklearn.preprocessing import LabelEncoder
Labelencoder_x=LabelEncoder()


# In[24]:


for i in range(0,5):
    x_train[:,i]=Labelencoder_x.fit_transform(x_train[:,i])
    
    


# In[25]:


x_train


# In[26]:


x_train[:,7]= Labelencoder_x.fit_transform(x_train[:,7])

x_train


# In[27]:


Labelencoder_y= LabelEncoder()
y_train= Labelencoder_y.fit_transform(y_train)

y_train


# In[28]:


for i in range(0,5):
    x_test[:,i]=Labelencoder_x.fit_transform(x_test[:,i])
    
x_test    
    


# In[29]:


x_test[:,7] = Labelencoder_x.fit_transform(x_test[:,7])

x_test


# In[30]:


Labelencoder_y= LabelEncoder()

y_test = Labelencoder_y.fit_transform(y_test)


y_test


# In[31]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train= ss.fit_transform(x_train)
x_test= ss.fit_transform(x_test)


# # Using Random Forest Classifier

# In[32]:


from sklearn.ensemble import RandomForestClassifier
RFClassifier= RandomForestClassifier()
RFClassifier.fit(x_train,y_train)


# In[33]:


from sklearn import metrics
y_pred = RFClassifier.predict(x_test)
print('Accuracy of random forest classifier is', metrics.accuracy_score(y_pred,y_test))


# # Using Decision Tree Classifier 

# In[34]:


from sklearn.tree import DecisionTreeClassifier
DTClassifier= DecisionTreeClassifier(criterion='entropy',random_state=0)
DTClassifier.fit(x_train,y_train)


# In[35]:


from sklearn import metrics
y_pred = DTClassifier.predict(x_test)


# In[36]:


print("accuracy of decision tree classifier is ", metrics.accuracy_score(y_pred,y_test))


# In[37]:


y_pred


# # Using Naive Bayes Classifier

# In[38]:


from sklearn.naive_bayes import GaussianNB
nb_clf= GaussianNB()
nb_clf.fit(x_train,y_train)


# In[39]:


y_pred= nb_clf.predict(x_test)
print('accuracy of naive bayes is', metrics.accuracy_score(y_pred,y_test))


# In[40]:


y_pred


# # Using KNN

# In[41]:


from sklearn.neighbors import KNeighborsClassifier
kn_clf=KNeighborsClassifier()
kn_clf.fit(x_train,y_train)


# In[42]:


y_pred= kn_clf.predict(x_test)
print("accuracy of KN is ",metrics.accuracy_score(y_pred,y_test))


# In[43]:


y_pred


# In[ ]:




