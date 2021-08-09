# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:14:54 2021
Link:
    https://betterprogramming.pub/anomaly-detection-with-isolation-forest-e41f1f55cc6
    
@author: 41162395
"""
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_iris


#https://stackoverflow.com/questions/62322882/load-iris-got-an-unexpected-keyword-argument-as-frame
import pandas as pd

data = load_iris()
X,y = data.data,data.target
df = pd.DataFrame(data.data, columns=data.feature_names)
df.head()

iforest = IsolationForest(n_estimators=100, max_samples='auto', 
                          contamination=0.05, max_features=4, 
                          bootstrap=False, n_jobs=-1, random_state=1)

pred= iforest.fit_predict(X)
df['scores']=iforest.decision_function(X)
df['anomaly_label']=pred

df[df.anomaly_label==-1]

df[df.anomaly_label==1]

df['anomaly']=df['anomaly_label'].apply(lambda x: 'outlier' if x==-1  else 'inlier')
fig=px.histogram(df,x='scores',color='anomaly')
fig.show()

fig = px.scatter_3d(df,x='petal width (cm)',
                       y='sepal length (cm)',
                       z='sepal width (cm)',
                       color='anomaly')
fig.show()

df[(df['petal width (cm)']<=0.13)|(df['petal width (cm)']>=2.46)]

df.describe().T
