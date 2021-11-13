# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 20:58:30 2021

@author: matth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('data.csv')
df = df.drop(['Unnamed: 32', 'id'], axis=1)

#df.shape # (569, 31)
df.head()
df.columns


df['diagnosis_num'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)


X = df[['concave points_worst', 'perimeter_worst', 'concave points_mean', 'radius_worst', 'perimeter_mean', 'area_worst',
               'radius_mean', 'area_mean', 'concavity_mean', 'concavity_worst', 'compactness_mean', 'compactness_worst', 
               'radius_se', 'perimeter_se', 'area_se', 'texture_worst', 'concave points_se', 'smoothness_mean', 'symmetry_mean',
                'fractal_dimension_worst']]    


drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','concave points_se','texture_worst']
x_1 = X.drop(drop_list1,axis = 1 )   




from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

y = df['diagnosis_num']

# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3, random_state=42)

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(x_train,y_train)

ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")





import pickle

pickl = {'model': clf_rf}
pickle.dump(pickl, open('model_file' + ".p", "wb"))


file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']
model.predict(x_test.iloc[1,:].values.reshape(1,-1))


list(x_test.iloc[1,:])







