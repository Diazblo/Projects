import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
from utils import dataset
import torchvision.transforms as transform
import torch
import numpy as np
from torch.autograd import Variable
# Fruit Classification with a CNN

from model import FNet


def test_mode(data,model):
     """Testing model on new data that the model have not seen  before
        Args:
            data: test data
            model : trained model
     """
     correct = 0
     total = 0
     for batch_idx1 , dat1 in enumerate(data):
                    data1 = dat1
                    inputs1 = data1[0].type(torch.FloatTensor)
                    labels1 = data1[1].type(torch.FloatTensor)
                    #GPU CHECK
                    if torch.cuda.is_available():
                        inputs1 = Variable(inputs1.cuda())
                    else:
                        inputs1 = Variable(inputs1)
                    
                    temp_out1 = model(inputs1)
                    
                    _,pred = torch.max(temp_out1,1)
                    total += labels1.size(0)
                    if torch.cuda.is_available():
                        correct+= ((pred.type(torch.FloatTensor)).cpu() == labels1.cpu()).sum()
                        
                    else:
                        correct+= (pred.type(torch.FloatTensor) == labels1).sum()
     #AVERAGE ACCURACY               
     acc = 100 * correct/total
     print('Test set  Accuracy ',acc)
        
def make_trainloader(train_dataset,test_dataset,validation_dataset,train_batch_size,test_batch_size):
    """This funtion will create iterable dataloaders for dataset
       Args:
           Self explainatory 
       Return:
           training set data loader , validation set dataloader and testset dataloader
    """
    trainloader = torch.utils.data.DataLoader(dataset=train_dataset ,batch_size =train_batch_size ,shuffle = True)
    testloader = torch.utils.data.DataLoader(dataset=test_dataset ,batch_size =test_batch_size ,shuffle = True)
    validationloader = torch.utils.data.DataLoader(dataset=validation_dataset ,batch_size =test_batch_size ,shuffle = True)
    
    return trainloader,testloader,validationloader

def data_preprocessing(train_set,test_set):
    """This funtion will perforn all preprocessing on data like splitting and applying transforms
       Args:
           dframe : main Dataframe
           train_set : training set
           test_set : testing set
       Returns:
           preprocessed dataset of train,test and validation
    """
    #Splitting data into validation and train set
    s1 = np.floor(0.8 * len(train_set)) #conversion as index won't take a float
    train_data = train_set.iloc[0:int(s1)]
    validation_data = train_set.iloc[int(s1):]
   
    #Resetting index so that whole dataframe can be used for iteration
    train_data = train_data.reset_index(drop=True)
    validation_data = validation_data.reset_index(drop=True)
    
    #Transformation:
    #   ToPiLImage :: will convert image to pil image while preserving range of pixels(0 -255)
    #   ToTensor   :: will convert every image pixel float number between range 0 -1
    transformations = transform.Compose([transform.ToPILImage(),transform.ToTensor()])
    
    #Calling our custom dataset and applying transforms
    train_dataset = dataset.ImageDataset(train_data ,transformations)
    test_dataset = dataset.ImageDataset(test_set ,transformations)
    validation_dataset = dataset.ImageDataset(validation_data,transformations)
    
    return train_dataset,test_dataset,validation_dataset

def forward_backprop(cost_fn,optimizer,model,inputs,labels,learning_rate):
            """This funtion will perforn forward pass and backward pass
               Args:
                   model: neural network model
                   inputs : dataset
                   labels : targets
                   laerning_rate : hyperparameter
               Return:
                   temp_out : output from forward pass
                   loss : loss calculated from loss funtion
                   cost_fn : loss funtion
            """
           
            #Taking out a temprary output from model's forward pass
            temp_out = model(inputs)
            #Resetting the current gradients for new gradients calculation
            optimizer.zero_grad()
            
            loss = cost_fn(temp_out , labels.type(torch.LongTensor))
            #back prop
            loss.backward()
            
            #Updating parameters
            optimizer.step()
            return model,temp_out,loss

def training_accuracy(temp_out,labels,loss):
    """Computes training accuracy and loss"""
    total1=0
    correct1=0
    _,pred = torch.max(temp_out,1)
    total1 += labels.size(0)
    if torch.cuda.is_available():
        correct1+= ((pred.type(torch.FloatTensor)).cpu() == labels.cpu()).sum()
        
    else:
        correct1+= (pred.type(torch.FloatTensor) == labels).sum()
        #Average accuracy        
        acc1 = 100 * correct1/total1
        print('Training loss is ',loss.data[0],'Training accuracy is ',acc1[0])

def validating(model,validationloader,cost_fn):
    """Computes accuracy and loss of validation set"""
    correct = 0
    total = 0
    for batch_idx1 , dat1 in enumerate(validationloader):
        data1 = dat1
        inputs1 = data1[0].type(torch.FloatTensor)
        labels1 = data1[1].type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs1 = Variable(inputs1.cuda())
        else:
            inputs1 = Variable(inputs1)
            
        temp_out1 = model(inputs1)
        loss1 = cost_fn(temp_out1 , labels1.type(torch.LongTensor))
        _,pred = torch.max(temp_out1,1)
        total += labels1.size(0)
        if torch.cuda.is_available():
            correct+= ((pred.type(torch.FloatTensor)).cpu() == labels1.cpu()).sum()
            
        else:
            correct+= (pred.type(torch.FloatTensor) == labels1).sum()
            
    acc = 100 * correct/total
    print('Loss on valdation set ',loss1.data[0],'  Accuracy ',acc[0])

                
def train_model(dataset_path, debug=False, destination_path='', save=False):
    """Training of model on train data set and validation on seperate dataset
       Args:
           dataset_path: path of dataset folder with images
           debug : If true code will print loss and accuracy of training set after each iteration
           destination path : path of csv file containing full image path with thier labels
           save : if true it will save whole model state.
    """
    #Loading dataframe
    dframe,train_set,test_set = dataset.create_and_load_meta_csv_df(dataset_path,destination_path,True,0.8)	
    
    train_dataset,test_dataset,validation_dataset = data_preprocessing(train_set,test_set)
 
    trainloader,testloader,validationloader = make_trainloader(train_dataset,test_dataset,validation_dataset,10,1)
    
    #temprory variables to store data and labels 
    data = []
    labels = torch.FloatTensor()
    inputs = torch.FloatTensor()
    
    #initialize model with hyper parameters
    # in_channel = 1 as image was converted to gray scale
    # kernel are for filter size
    # outchannels are for total number for kernels stack
    #Stride has been set to 1
    model = FNet(in_channel=1,out_channels1=16,out_channels2=32,kernel1=5,pool_kernel=2,stride=1)
    
    if torch.cuda.is_available():
        model.cuda()
    
    n_epochs  = 3
    cost_fn = nn.CrossEntropyLoss()
    learning_rate = 0.005
    #We used Adam optimizer of SGD, as it can maintain adaptive learning rates for diffrent parameters that is estimated from first and second moment of gradients
    #We set the weight decay for L2 norm
    optimizer  = torch.optim.Adam(model.parameters() , lr  = learning_rate,weight_decay=1e-4)
    
    for epochs in range(n_epochs):
        for batch_idx , dat in enumerate(trainloader):
            data = dat
            inputs = data[0].type(torch.FloatTensor)
            labels = data[1].type(torch.FloatTensor)
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:    
                inputs = Variable(inputs)
                labels = Variable(labels)
            
            model,temp_out,loss =  forward_backprop(cost_fn,optimizer,model,inputs,labels,learning_rate)
            
            if (debug==True):
                training_accuracy(temp_out,labels,loss)
                
        #This validation set will be tested after each epoch        
        validating(model,validationloader,cost_fn)
    
    #Saving whole model    
    if(save == True):
        torch.save(model,f = 'model_beta.pt')
    #Testing model on new test data
    test_mode(testloader,model)
    
   

if __name__ == "__main__":
	train_model(dataset_path='../Data/fruits/', debug=False,save=True, destination_path='./')
