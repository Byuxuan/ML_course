import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


dataset=pd.read_csv('data/white.csv',sep=';')
#dataset=pd.read_csv('data/white.csv',sep=';')



dataset['label']=dataset.quality.apply(lambda x: 1 if x  > 5 else 0)

dataset.to_csv('white_new.csv',index=False)
