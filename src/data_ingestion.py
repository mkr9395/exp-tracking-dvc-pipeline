import pandas as pd 
import numpy as np 

import pathlib
import os


from sklearn.model_selection import train_test_split

url = 'https://raw.githubusercontent.com/campusx-official/toy-datatsets/main/student_performance.csv'


df = pd.read_csv(url)

train, test = train_test_split(df,test_size=0.2, random_state=42)


train.to_csv('./data/raw/train.csv',index=False)
test.to_csv('./data/raw/test.csv',index=False)