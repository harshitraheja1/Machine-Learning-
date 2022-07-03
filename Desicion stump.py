# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:10:13 2022

@author: Lenovo
"""

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas
data = {'CGPA':['g9','l9','g9','l9','g9'],
        'Inter':['Y','N','N','N','Y'],
        'PK':['++','==','==','==','=='],
        'CS':['G','G','A','A','G'],
        'Job':['Y','Y','N','N','Y']}
table=pandas.DataFrame(data,columns=["CGPA","Inter","PK","CS","Job"])
encoder=LabelEncoder()
for i in table:
    table[i]=encoder.fit_transform(table[i])
X=table.iloc[:,0:4].values
t=table.iloc[:,4].values
X_train,X_test,t_train,t_test=train_test_split(X,t,test_size=0.1,random_state=2)
print("\nDecision Tree Classifier")
model = tree.DecisionTreeClassifier()
model = model.fit(X_train, t_train)
predicted_value = model.predict(X_test)
print(predicted_value)
tree.plot_tree(model)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(t_test, predicted_value)
print("Accuracy: ",accuracy)