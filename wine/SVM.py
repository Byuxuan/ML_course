import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


#dataset=pd.read_csv('data/red.csv',sep=';')
dataset=pd.read_csv('data/white_new.csv')

print(dataset.columns)
print(dataset.isnull().any())
#print(dataset.columns)

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

'''
linear_clf=SVC(random_state=0,tol=1e-5,kernel='linear')
linear_clf.fit(x_train,y_train)

y_pred_LinearSVC=linear_clf.predict(x_test)

print(classification_report(y_test,y_pred_LinearSVC,digits=4))



rbf_clf=SVC(random_state=0,tol=1e-5,kernel='rbf')
rbf_clf.fit(x_train,y_train)

y_pred_rbfSVC=rbf_clf.predict(x_test)

print(classification_report(y_test,y_pred_rbfSVC,digits=4))
'''
def svm_cross_validation(train_x, train_y):    
  
    model = SVC(kernel='rbf', probability=True)    
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100], 'gamma': [0.125, 0.25, 0.5 ,1, 2, 4]}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1,scoring='f1_micro')    
    grid_search.fit(train_x, train_y)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(train_x, train_y)    
    return model

model=svm_cross_validation(x_train, y_train)
y_pred_rbfSVC=model.predict(x_test)

print(classification_report(y_test,y_pred_rbfSVC,digits=4))
























