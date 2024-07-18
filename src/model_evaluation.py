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


# load parametrs from logging

with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)


# calculate metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
    
# log metrics (exp tracking) and parameters using dvclive

with Live(save_dvc_exp=True) as live:
    live.log_metric('accuracy', accuracy)
    live.log_metric('precision', precision)
    live.log_metric('recall', recall)
    live.log_metric('f1', f1)

    for param, value in params.items():
        for key, val in value.items():
            live.log_param(f'{param}_{key}', val)
            

# Save the metrics to a JSON file for comparability with DVC
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1
}

try:
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
except TypeError as e:
    print("Serialization error:", e)
    print("Metrics dictionary contains non-serializable values.")