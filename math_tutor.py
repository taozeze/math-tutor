
# coding: utf-8

# In[1]:

import pandas as pd
import csv as csv


# In[2]:

df = pd.read_csv('algebra_2005_2006/algebra_2005_2006_train.txt', sep='\t')


# In[3]:

df.head(100)


# In[4]:

correct = df[df['Problem View'] == 1]


# In[5]:

df.groupby(['Anon Student Id']).count()


# In[ ]:




# In[16]:

from sklearn import tree
X = df.columns[16:17] 
y = df['Correct First Attempt'] 
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)


# In[ ]:



