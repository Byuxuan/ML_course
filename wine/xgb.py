import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import xgboost as xgb



dataset=pd.read_csv('data/white_new.csv')
#dataset=pd.read_csv('data/red.csv',sep=';')

print(dataset.columns)
print(dataset.isnull().any())
#print(dataset.columns)
dataset.quality=dataset.quality-3
quality=dataset.groupby(['quality']).agg({"density": lambda x: len(x)}).reset_index()

quality.columns=['quality','count']
print(quality)

dataset=np.array(dataset)

dataset_x=dataset[:,:-2]
dataset_y=dataset[:,-1]

x_train, x_test, y_train, y_test = train_test_split(dataset_x,dataset_y,test_size = 0.3)



scaler=StandardScaler().fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

dataset_train=xgb.DMatrix(x_train,label=y_train)
#dataset2=xgb.DMatrix(dataset_valid_x,label=dataset_valid_y)
dataset_test=xgb.DMatrix(x_test)

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 多分类的问题
    'num_class':2,
    'gamma': 0.1,                  
    'max_depth': 12,               # 构建树的深度，越大越容易过拟合
    'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,              # 随机采样训练样本
    'colsample_bytree': 0.7,       # 生成树时进行的列采样
    'min_child_weight': 3,
    'silent': 0,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.001,                  # 如同学习率
    'seed': 1000,
    'nthread': 4,                  # cpu 线程数
    'early_stopping_rounds':20
}

watchlist = [(dataset_train,'train')]
model = xgb.train(params,dataset_train,num_boost_round=10000,evals=watchlist)

y_pred=model.predict(dataset_test)

print(classification_report(y_test,y_pred,digits=4))

