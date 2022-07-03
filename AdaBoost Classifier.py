# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:10:19 2022

@author: Lenovo
"""

import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
data = {'CGPA':['g9','l9','g9','l9','g9'],
        'Inter':['Y','N','N','N','Y'],
        'PK':['++','==','==','==','=='],
        'CS':['G','G','A','A','G'],
        'Job':['Y','Y','N','N','Y']}
table=pandas.DataFrame(data,columns=["CGPA","Inter","PK","CS","Job"])
table.where(table["CGPA"]=="g9").count()
encoder=LabelEncoder()
for i in table:
    table[i]=encoder.fit_transform(table[i])
X=table.iloc[:,0:4].values
t=table.iloc[:,4].values
X_train,X_test,t_train,t_test=train_test_split(X,t,test_size=0.2,random_state=2)
model = AdaBoostClassifier(n_estimators=3)
model.fit(X_train,t_train)
if model.predict([[0,1,1,1]])==1:
    print("Got JOB")
else:
    print("Didn't get JOB")