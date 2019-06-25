import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.model_selection import GridSearchCV



dataset=pd.read_csv('data/adult.csv')



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
def tree_validation(train_x, train_y):    
  
    model = tree.DecisionTreeClassifier(criterion='entropy')    
    param_grid = {'criterion':['entropy','gini'],'max_depth': [5,6,7,8,9,10,11,12],'max_features':[11,'auto','sqrt','log2']}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1,scoring='f1_micro')    
    grid_search.fit(train_x, train_y)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
    model = tree.DecisionTreeClassifier(criterion=best_parameters['criterion'],max_depth=best_parameters['max_depth'],max_features=best_parameters['max_features'] )    
    model.fit(train_x, train_y)    
    return model

model=tree_validation(x_train, y_train)
y_pred=model.predict(x_test)

print(classification_report(y_test,y_pred,digits=4))

























