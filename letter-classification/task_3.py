import pandas as pd
from torch import optim
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
import numpy as np

data = pd.read_csv('letter-recognition.data',header=None)

#Seperating labels and features
x = data.iloc[:,1:]
y = data.iloc[:,0]

#Converting labels to integer representation
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform(y)



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


#Logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)

print(lr.score(x_test,y_test))


#Support vector Machine
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

print(svc.score(x_test,y_test))


#K Nearest Classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)

print(knn.score(x_test,y_test))




#Building a neural net
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(16, 32)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.hidden2 = nn.Linear(32,64)
        self.output = nn.Linear(64, 26)
        
  
    def forward(self, x):
        x = self.hidden1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.output(x)
        return x


#initating model and loss funtion
model = Model()
loss_function = nn.CrossEntropyLoss()

data = pd.read_csv('letter-recognition.data',header=None)

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
data.iloc[:,0] = lb.fit_transform(data.iloc[:,0])

data = torch.tensor(data.values)

trainloader = DataLoader(data[:int(len(data)*0.7)],batch_size=50)
validloader = DataLoader(data[int(len(data)*0.7):],batch_size=50)


#Accuracy check
def acc(validloader,model):
    accuracy =0
    model.eval()
    for data in validloader:
        data_input = (data[:,1:]).type(torch.FloatTensor)
        target = data[:,0]
        temp_out = model(data_input)
        for i in range(len(temp_out)):
            if(target[i] == torch.argmax(temp_out[i])):
                accuracy+=1
    
    return accuracy/len(validloader.dataset)
 
    
lr=0.01

for epoch in range(1, 101):
    if(epoch%40 ==0):
        lr = lr*0.6
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss, valid_loss = [], []
    model.train()
    for data in trainloader:
        data_input = (data[:,1:]).type(torch.FloatTensor)
        target = data[:,0]
        optimizer.zero_grad()
      ## 1. forward propagation
        output = model(data_input)
        
        ## 2. loss calculation
        loss = loss_function(output, target)
        
        ## 3. backward propagation
        loss.backward()
        
        ## 4. weight optimization
        optimizer.step()
        
        train_loss.append(loss.item())
        
    ## evaluation part 
    model.eval()
    for data in validloader:
        data_input = (data[:,1:]).type(torch.FloatTensor)
        target = data[:,0]
        output = model(data_input)
        loss = loss_function(output, target)
        valid_loss.append(loss.item())

    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))
    if(epoch%10==0):
        print(end="\n\n")
        print(epoch,acc(trainloader,model))
        print(acc(validloader,model),end='\n\n')





