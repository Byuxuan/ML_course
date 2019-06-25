import pandas as pd
import numpy as np

dataset=pd.read_csv('data/adult.data')
dataset.columns=['age','workclass','fnlwgt','education','education_num','married','occupation','relation','race','sex','capital_gain','capital_loss','hours_per_week','native','label']

'''
print(dataset.workclass.unique())

print(dataset.education.unique())

print(dataset.married.unique())

print(dataset.occupation.unique())

print(dataset.relation.unique())

print(dataset.race.unique())

print(dataset.sex.unique())

print(dataset.native.unique())

print(dataset.label.unique())

print(dataset.isnull().any())
'''
workclass={}
education={}
married={}
occupation={}
relation={}
race={}
sex={}
native={}
label={}

dataset=np.array(dataset)

index=0
for data  in dataset[:,1]:

	if data not in workclass:
	
		workclass[data]=index
		index=index+1



index=0
for data  in dataset[:,3]:
        if data not in education:
                education[data]=index
                index=index+1

index=0
for data  in dataset[:,5]:
        if data not in married:
                married[data]=index
                index=index+1

index=0
for data  in dataset[:,6]:
        if data not in occupation:
                occupation[data]=index
                index=index+1

index=0
for data  in dataset[:,7]:
        if data not in relation:
                relation[data]=index
                index=index+1
index=0
for data  in dataset[:,8]:
        if data not in race:
                race[data]=index
                index=index+1
index=0
for data  in dataset[:,9]:
        if data not in sex:
                sex[data]=index
                index=index+1
index=0
for data  in dataset[:,-2]:
        if data not in native:
                native[data]=index
                index=index+1
index=0
for data  in dataset[:,-1]:
        if data not in label:
                label[data]=index
                index=index+1


for  data in dataset:
	
	data[1]=workclass[data[1]]
	data[3]=education[data[3]]
	data[5]=married[data[5]]
	data[6]=occupation[data[6]]
	data[7]=relation[data[7]]
	data[8]=race[data[8]]
	data[9]=sex[data[9]]
	data[-2]=native[data[-2]]	
	data[-1]=label[data[-1]]

dataset=pd.DataFrame(dataset)
dataset.columns=['age','workclass','fnlwgt','education','education_num','married','occupation','relation','race','sex','capital_gain','capital_loss','hours_per_week','native','label']
#dataset.to_csv('adult.csv',index=False)

workclasscols = pd.get_dummies(dataset['workclass'])
workclasscols.columns = ['workclass'+str(i) for i in range(1,10)]
dataset = pd.concat([dataset,workclasscols],axis=1)

educationcols = pd.get_dummies(dataset['education'])
educationcols.columns = ['education'+str(i) for i in range(1,17)]
dataset = pd.concat([dataset,educationcols],axis=1)

marriedcols = pd.get_dummies(dataset['married'])
marriedcols.columns = ['married'+str(i) for i in range(1,8)]
dataset = pd.concat([dataset,marriedcols],axis=1)

occupationcols = pd.get_dummies(dataset['occupation'])
occupationcols.columns = ['occupation'+str(i) for i in range(1,16)]
dataset = pd.concat([dataset,occupationcols],axis=1)

relationcols = pd.get_dummies(dataset['relation'])
relationcols.columns = ['relation'+str(i) for i in range(1,7)]
dataset = pd.concat([dataset,relationcols],axis=1)

racecols = pd.get_dummies(dataset['race'])
racecols.columns = ['race'+str(i) for i in range(1,6)]
dataset = pd.concat([dataset,racecols],axis=1)

nativecols = pd.get_dummies(dataset['native'])
nativecols.columns = ['native'+str(i) for i in range(1,43)]
dataset = pd.concat([dataset,nativecols],axis=1)

dataset['label2']=dataset.label


dataset=dataset.drop(['workclass','education','married','occupation','relation','race','native','label'],axis=1)
dataset.to_csv('data/adult.csv',index=False)






















































