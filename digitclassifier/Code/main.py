import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")



from nnet import model

device = "cuda:0" if torch.cuda.is_available() else "cpu"
trn_BATCH_SIZE = 4
test_BATCH_SIZE = 100



trainloader = torch.utils.data.DataLoader(datasets.MNIST('mnist_data', 
                                                          download=True, 
                                                          train=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                              transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                          ])),
                                           
                                           batch_size=trn_BATCH_SIZE, 
                                           shuffle=True)




# download and transform test dataset
test_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist_data', 
                                                          download=True, 
                                                          train=False,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                              transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                          ])),  
                                           
                                           batch_size=test_BATCH_SIZE, 
shuffle=True)

# NOTE: Don't change these settings
# Layer size
N_in = 28 * 28 # Input size
N_h1 = 256 # Hidden Layer 1 size
N_h2 = 256 # Hidden Layer 2 size
N_out = 10 # Output size
# Learning rate
lr = 0.001


# init model
net = model.FullyConnected(N_in, N_h1, N_h2, N_out, device=device)

# TODO: Define number of epochs
N_epoch = 2 


''' Instead of slipting validation set from training set we set a condition check 
    to auto filter data for validation set we set check to 5 as the total iteration in 
    one epoch will be (60000/4) = 15000 and every 5th batch will be a validation batch 
    so total training dataset is been split into 80/20 ratio with validation set as 12000 and training as 48000 dataset'''
loss_train = []
loss_valid = []
acc_train = []
acc_valid = []
temp_idx = 0
temp_idx1 = 0
for i in range(N_epoch):
    j = 0
    
    for batch_idx, (x, target) in enumerate(trainloader):
        j+=1
        x, target = x, target
        x = x.reshape(trn_BATCH_SIZE,-1)
        if (j%5!=0):
            temp_idx+=1
            creloss, accuracy, outputs = net.train(x, target, lr=0.01,debug=False)
            if(temp_idx%1000 == 0):
                loss_train.append(creloss)
                acc_train.append(accuracy)
    j=0
    k=0
    
    """Validation Set"""
    validation_loss = validation_accuracy = 0
    for batch_idx, (x, target) in enumerate(trainloader):
        j+=1
        x, target = x, target
        x = x.reshape(trn_BATCH_SIZE,-1)
        if (j%5==0):
            k+=1
            temp_idx1+=1
            valid_loss,valid_acc,outputs = net.eval(x,target)
            validation_loss+=valid_loss
            validation_accuracy+=valid_acc
            if(temp_idx1%400 == 0):
                loss_valid.append(valid_loss)
                acc_valid.append(valid_acc)
            
    print('Epoch: ',i+1,' Validation loss ',validation_loss/k,' accuracy: ',validation_accuracy/k)
    


accuracy_test = 0
j = 0
for batch_idx, (x, target) in enumerate(test_loader):
    j+=1
    x, target = x, target
    x = x.reshape(test_BATCH_SIZE,-1)
    score,idx = net.predict(x)
    accuracy_test += torch.sum( idx == target )
print('Test set accuray ',accuracy_test/j,'%')
