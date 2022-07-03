# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:07:31 2022

@author: Lenovo
"""

import numpy as np
from sklearn.datasets import make_classification 

x, t = make_classification(100, 5, n_classes = 3, random_state= 50, n_informative = 3, n_clusters_per_class = 1)

from sklearn.model_selection import train_test_split
x_train,x_test, t_train , t_test = train_test_split(x,t,test_size = 0.3 )

from sklearn import svm 
model = svm.SVC(kernel='linear')

model.fit(x_train,t_train)
 
y_pred = model.predict(x_test)
from sklearn import metrics 
score = metrics.accuracy_score(t_test,y_pred)
print(score)
#%%
import matplotlib.pyplot as plt
w = model.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-3, 3)
yy = a *xx - model.intercept_[0]/w[1]
h0 = plt.plot(xx,yy,'k-',label="non weighted div" )
plt.scatter(x_train[:,0],x_train[:,1],c=t_train)
plt.show()
#%%
import matplotlib.pyplot as plt
plt.scatter(x_train[:,0],x_train[:,1],c=t_train)
plt.show()