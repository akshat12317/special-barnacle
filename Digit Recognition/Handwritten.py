import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv("train.csv").as_matrix()
clf=RandomForestClassifier()#Empty classifier

##Splitting the data set

##Training dataset list of features=intensity value and corrosponding labels=digit
xtrain=data[0:21000,1:] #training data from 1st column
trainlabel=data[0:21000,0]

clf.fit(xtrain,trainlabel)  #training the classifier


#Testing data

xtest=data[21000:,1:]  #testing the data from 1st column
actual_label=data[21000:,0]  #used to checking the accurcy


#Testing the accurecy
prediction=clf.predict(xtest)
count=0

for i in range(0,21000):
    if prediction[i]==actual_label[i]:
        count+=1
    else: 
        count+=0
result=((count/21000)*100)
print(f'Accuracy={result}')


