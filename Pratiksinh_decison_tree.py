# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 12:09:15 2022

@author: Pratik
"""

import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import graphviz 
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


data_pratiksinh = pd.read_csv("C:/Users/user/Downloads/student-por.csv", sep = ';')

    
data_pratiksinh.head()

print(data_pratiksinh.info())

print(data_pratiksinh.isnull().sum())

print(data_pratiksinh.describe())

data_pratiksinh.dtypes

#statistical calculation
print(data_pratiksinh.describe().T)



data_pratiksinh.nunique()

#Correlation
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(17, 15))
corr_mask = np.triu(data_pratiksinh.corr())
h_map = sns.heatmap(data_pratiksinh.corr(), mask=corr_mask, annot=True, cmap='Blues')
plt.yticks(rotation=360)
plt.xticks(rotation=90)
plt.show()

data_pratiksinh.hist(bins=70, figsize=(12,10))


data_pratiksinh['total grade'] = (data_pratiksinh['G1']+data_pratiksinh['G2']+data_pratiksinh['G3'])

data_pratiksinh['total grade']

pass_pratiksinh = []
for value in data_pratiksinh["total grade"]:
    if value>=35:
        pass_pratiksinh.append(1)
    else:
        pass_pratiksinh.append(0)
        
pass_pratiksinh        
        
data_pratiksinh["pass_pratiksinh"]= pass_pratiksinh

data_pratiksinh.info()

data_pratiksinh = data_pratiksinh.drop(['G1','G2','G3',"total grade"],axis=1)

data_pratiksinh.info()



data_pratiksinh.nunique()


features_pratiksinh=data_pratiksinh.drop(['pass_pratiksinh', 'address' ],axis=1)
target_pratiksinh=data_pratiksinh["pass_pratiksinh"]

data_pratiksinh.value_counts()

#separate the numeric and categorical features
numeric_features_pratiksinh = features_pratiksinh.select_dtypes(include=['int64'])
print(numeric_features_pratiksinh.head())
cat_features_pratiksinh = features_pratiksinh.select_dtypes(include=['object'])
print(cat_features_pratiksinh.head())

#Transformer 
transformer_pratiksinh = ColumnTransformer([('encoder', OneHotEncoder(), cat_features_pratiksinh.columns)], remainder='passthrough')

#decision tree classifier
clf_pratiksinh = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=60)

#pipeline
pipeline_pratiksinh = Pipeline([("trans", transformer_pratiksinh),("clf",clf_pratiksinh)])

#Splitting the data
X_train_pratiksinh,X_test_pratiksinh, Y_train_pratiksinh,Y_test_pratiksinh = train_test_split(features_pratiksinh,target_pratiksinh,test_size=0.2, random_state=63, shuffle=True)




#Building Model

train_pipeline = pipeline_pratiksinh.fit(X_train_pratiksinh,Y_train_pratiksinh)

import random
random.seed(63)

strat_k_fold = StratifiedKFold(n_splits=10, shuffle=True,random_state=63)
crossv_score = cross_val_score(pipeline_pratiksinh,X_train_pratiksinh, Y_train_pratiksinh, scoring='accuracy',cv=strat_k_fold)
print("cross validation-",crossv_score)
print("mean of 10",crossv_score.mean())

#Graphviz
import os
os.environ["PATH"] += os.pathsep + '"E:/Anaconda/Lib'

tree.plot_tree(clf_pratiksinh)
graphviz_pratiksinh = tree.export_graphviz(train_pipeline.steps[1][1],
                             class_names=['Fail','Pass'],  
                             out_file=None, filled=True, 
                             rounded=True,  
                      special_characters=False)
graph = graphviz.Source(graphviz_pratiksinh)
graph.render('DecisionTree', format = 'png')


#accuracy training
Y_train_pred = pipeline_pratiksinh.predict(X_train_pratiksinh)
accuracyScore = accuracy_score(Y_train_pratiksinh,Y_train_pred)
print("Training accuracy ",accuracyScore)

#accuracy testing
Y_test_pred = pipeline_pratiksinh.predict(X_test_pratiksinh)
accuracyScore = accuracy_score(Y_test_pratiksinh,Y_test_pred)
print("Testing accuracy ",accuracyScore)

#report
print('Confusion matrix using testing-\n' ,confusion_matrix(Y_test_pratiksinh,Y_test_pred))
print('Precision-\n', precision_score(Y_test_pratiksinh, Y_test_pred))
print('Recall-\n', recall_score(Y_test_pratiksinh, Y_test_pred))
print('Accuracy-\n', accuracy_score(Y_test_pratiksinh, Y_test_pred))


#Tuning...........

#parametres defined for randomGrid search
parameters={'clf__min_samples_split' : range(10,300,20),'clf__max_depth':range(1,30,2),'clf__min_samples_leaf':range(1,15,3)}
    
#random grid search 
random_grid_search=RandomizedSearchCV(estimator = pipeline_pratiksinh,
                  scoring='accuracy',cv=5,
                   param_distributions=parameters,
                   n_iter=7,refit = True, verbose = 3)
    
random_grid_search.fit(X_train_pratiksinh, Y_train_pratiksinh)


#best parameters  
best_parameter=random_grid_search.best_params_
best_score=random_grid_search.best_score_
best_estimator=random_grid_search.best_estimator_

print ('The best param- \n', best_parameter)
print ('The best score- \n', best_score)
print ('The best estimator- \n', best_estimator)

#best estimator model
best_estimator_model = random_grid_search.best_estimator_
Y_grid_predict = best_estimator_model.predict(X_test_pratiksinh)

confusion_matrix = confusion_matrix(Y_test_pratiksinh, Y_grid_predict)
accuracy = accuracy_score(Y_test_pratiksinh, Y_grid_predict)
precision = precision_score(Y_test_pratiksinh, Y_grid_predict)
recall = recall_score(Y_test_pratiksinh, Y_grid_predict)



print('The confusion matrix for testing data-\n', confusion_matrix)
print('Precision-' , precision)
print('Recall-', recall)
print('Accuracy-', accuracy)



#save
import joblib
joblib.dump(best_estimator_model, "decision_tree_pratiksinh.pkl")
joblib.dump(pipeline_pratiksinh, "pipeline_pratiksinh.pkl")
