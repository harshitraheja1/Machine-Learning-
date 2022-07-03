# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:24:23 2022

@author: Lenovo
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import tree
X, t = make_classification(100, 5, n_classes = 3, random_state= 70, n_informative = 2, n_clusters_per_class = 1)
X_train, X_test, t_train, t_test = train_test_split(X,t)
#%%
model = tree.DecisionTreeClassifier(max_depth=3)
model.fit(X_train, t_train)
tree.plot_tree(model)
#%%

