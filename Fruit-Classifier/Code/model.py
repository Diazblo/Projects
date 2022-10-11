import torch.nn as nn
         
class FNet(nn.Module):
    """Fruit Net
        Created 2 convulution layer and one full connected layer with one hidden layer for whole network
        
    """
   
    def __init__(self,in_channel,out_channels1,out_channels2,kernel1,pool_kernel,stride):
        """Initializer of model
            Args:
                 in_channel = number of channels for image E.g:1 for grey and 3 for ''RGB
                 outchannel1 = total number of  filters for convulution layer 1
                 outchannel2 = total number of  filters for convulution layer 2
                 kernel1  =  size of single filter (kernel1 x kernel1)
                 pool_kernel = size of single filter in maxpool layers
                 Stride  = mean steps for convulution operations
        """
    
        super(FNet,self).__init__()
        
        #First convulution layer
        self.cnn1 = nn.Conv2d(in_channels = in_channel,out_channels = out_channels1,kernel_size=kernel1,stride=stride,padding=int((kernel1-1)/2))
        self.batch1 = nn.BatchNorm2d(out_channels1)
        self.relu1 = nn.ReLU()
        
        #First pooling layer
        #reducing image size
        self.maxpool1 = nn.MaxPool2d(kernel_size=pool_kernel)
        
        #Second convulution layer
        self.cnn2 = nn.Conv2d(in_channels = out_channels1,out_channels = out_channels2,kernel_size=kernel1,stride=stride,padding=int((kernel1-1)/2))
        self.batch2 = nn.BatchNorm2d(out_channels2)
        self.relu2 = nn.ReLU()
        
        #Second pooling layer
        #reducing image size
        self.maxpool2 = nn.MaxPool2d(kernel_size=pool_kernel)
        
        #Fully connected layer
        #Input Dimension is 100
        #Output dimension after first pooling  = 100/2 = 50
        #Output dimension after second convulution = 50(padding)
        #Output dimension after second max pooling is 50/2 =  25
        #Final input dimension for FCL will be out_channels2 * 25 * 25
        # 5 is the number of classes
        self.fc1 = nn.Linear(out_channels2 * 25 * 25 , 5)
        
      
    def forward(self, x):
        # forward propagation
        out = self.cnn1(x)
        out = self.batch1(out)
        out = self.relu1(out)
        
        out = self.maxpool1(out)
        
        out = self.cnn2(out)
        out = self.batch2(out)
        out = self.relu2(out)
        
        out = self.maxpool2(out)
        
        #resizing our tensor for compactibility for fully connected layer
        out = out.view(out.size(0) , -1)
        
        out = self.fc1(out)
        
        return out
        
    
    
    
if __name__ == "__main__":
    net = FNet()
