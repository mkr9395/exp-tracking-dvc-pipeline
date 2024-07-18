import pandas as pd
import numpy as np

import os
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import dvclive

from dvclive import Live
import json

test = pd.read_csv('./data/processed/test_processed.csv')

X_test = test.drop(columns =['Placed'])
y_test  = test['Placed']  

# load the model 
rf = pickle.load(open('model.pkl','rb'))

y_pred= rf.predict(X_test)

# calculate metrics
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)


# load parametrs from logging

with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)
    
# log metrics (exp tracking) and parameters using dvclive

with Live(save_dvc_exp=True) as live:
    live.log_metric('accuracy',accuracy_score(y_test,y_pred))
    live.log_metric('precision', precision_score(y_test,y_pred))
    live.log_metric('recall', recall_score(y_test,y_pred))
    live.log_metric('f1', f1_score(y_test,y_pred))

    for param, value in params.items():
        for key, value in value.items():
            live.log_param(f'{param}_{key}', val)
            

# save the metrics to a json file for comparibility with DVC

metrics = {
    'accuracy' : accuracy,
    'precision' : precision,
    'recall' : recall,
    'f1_score' : f1_score
}

with open('metrics.json','w') as f:
    json.dump(metrics, f, indent = 4)