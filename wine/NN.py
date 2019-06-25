import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from sklearn.metrics import accuracy_score



'''
class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer,batch_first=True)
        self.relu=nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, n_class)


    def forward(self, x):
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1),
                                #   self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1),
                               #   self.hidden_dim)).cuda()
        x=x.unsqueeze(0)

        #print(type(x))
        out, _ = self.lstm(x)
        out=self.relu(out)
        out=out.squeeze(0)
        out = self.classifier(out)
        return out

'''
class Full_Net(nn.Module):
    """
    在上面的Activation_Net的基础上，增加了一个加快收敛速度的方法——批标准化
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3,n_class):
        super(Full_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_class))
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x






#print(len(x_train[0]))
'''
def LSTM(x_train,x_test,y_train,y_test):
	model=Rnn(len(x_train[0]),512,3,7).cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-3)


	x_train=torch.FloatTensor(x_train).cuda()
	y_train=torch.LongTensor(y_train).cuda()




	for epoch in range(1000):

        	output=model(x_train)
        	loss=criterion(output,y_train)
        	loss.backward()
        	optimizer.step()
        	print(epoch,":",loss.cpu().detach().numpy())

	x_test=torch.FloatTensor(x_test).cuda()
	y_pred=model(x_test)
	_,y_pred=y_pred.data.topk(1)
	y_pred=y_pred.cpu().numpy()
	print(classification_report(y_test,y_pred,digits=4))
'''
def Full(x_train,x_test,y_train,y_test):
        model=Full_Net(len(x_train[0]),512*2,512*2,512*2,2).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.00001,weight_decay=1e-3)


        x_train=torch.FloatTensor(x_train).cuda()
        y_train=torch.LongTensor(y_train).cuda()




        for epoch in range(50000):

                output=model(x_train)
                loss=criterion(output,y_train)
                loss.backward()
                optimizer.step()
                if epoch%10000==0:
                    print(epoch,":",loss.cpu().detach().numpy())

        x_test=torch.FloatTensor(x_test).cuda()
        y_pred=model(x_test)
        _,y_pred=y_pred.data.topk(1)
        y_pred=y_pred.cpu().numpy()
        print(classification_report(y_test,y_pred,digits=4))
        print(accuracy_score(y_test,y_pred))


if __name__=='__main__':
	dataset=pd.read_csv('data/white_new.csv')
#dataset=pd.read_csv('data/red.csv',sep=';')

	print(dataset.columns)
	print(dataset.isnull().any())
	dataset.quality=dataset.quality-3

	quality=dataset.groupby(['quality']).agg({"density": lambda x: len(x)}).reset_index()

	quality.columns=['quality','count']
	print(quality)

	dataset=np.array(dataset)

	dataset_x=dataset[:,:-2]
	dataset_y=dataset[:,-1]

	x_train, x_test, y_train, y_test = train_test_split(dataset_x,dataset_y,test_size = 0.3)
	
	Full(x_train,x_test,y_train,y_test)
	x_test=x_train[:1000]
	y_test=y_train[:1000]
	Full(x_train,x_test,y_train,y_test)

