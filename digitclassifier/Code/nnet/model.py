import math
import torch

class FullyConnected:
    """Constructs the Neural Network architecture.

    Args:
        N_in (int): input size
        N_h1 (int): hidden layer 1 size
        N_h2 (int): hidden layer 2 size
        N_out (int): output size
        device (str, optional): selects device to execute code. Defaults to 'cpu'
    
    Examples:
        >>> network = model.FullyConnected(2000, 512, 256, 5, device='cpu')
        >>> creloss, accuracy, outputs = network.train(inputs, labels)
    """

    def __init__(self, N_in, N_h1, N_h2, N_out, device='cpu'):
        """Initializes weights and biases, and construct neural network architecture.
        
        One [recommended](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) approach is to initialize weights randomly but uniformly in the interval from [-1/n^0.5, 1/n^0.5] where 'n' is number of neurons from incoming layer. For example, number of neurons in incoming layer is 784, then weights should be initialized randomly in uniform interval between [-1/784^0.5, 1/784^0.5].
        
        You should maintain a list of weights and biases which will be initalized here. They should be torch tensors.

        Optionally, you can maintain a list of activations and weighted sum of neurons in a dictionary named Cache to avoid recalculation of those. If tensors are too large it could be an issue.
        """
        self.N_in = N_in
        self.N_h1 = N_h1
        self.N_h2 = N_h2
        self.N_out = N_out

        self.device = torch.device(device)

        w1 = torch.Tensor(N_h1,N_in).uniform_(-1/math.pow(N_in,0.5), 1/math.pow(N_in,0.5)).float()
        w2 = torch.Tensor(N_h2,N_h1).uniform_(-1/math.pow(N_h1,0.5), 1/math.pow(N_h1,0.5)).float()
        w3 = torch.Tensor(N_out,N_h2).uniform_(-1/math.pow(N_h2,0.5), 1/math.pow(N_h2,0.5)).float()
        self.weights = {'w1': w1, 'w2': w2, 'w3': w3}
              
        b1 = torch.Tensor(N_h1).uniform_(-1/math.pow(N_in,0.5), 1/math.pow(N_in,0.5)).float()
        b2 = torch.Tensor(N_h2).uniform_(-1/math.pow(N_h1,0.5), 1/math.pow(N_h1,0.5)).float()
        b3 = torch.Tensor(N_out).uniform_(-1/math.pow(N_h2,0.5), 1/math.pow(N_h2,0.5)).float()
        self.biases = {'b1': b1, 'b2': b2, 'b3': b3}
        
        self.cache = {'z1': 0, 'z2': 0, 'z3': 0,'a1':0, 'a2' : 0 , 'a3' : 0}

    def train(self, inputs, labels, lr=0.001, debug=False):
        """Trains the neural network on given inputs and labels.

        This function will train the neural network on given inputs and minimize the loss by backpropagating and adjusting weights with some optimizer.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 
            labels (torch.tensor): correct labels. Size (batch_size)
            lr (float, optional): learning rate for training. Defaults to 0.001
            debug (bool, optional): prints loss and accuracy on each update. Defaults to False

        Returns:
            creloss (float): average cross entropy loss
            accuracy (float): ratio of correctly classified to total samples
            outputs (torch.tensor): predictions from neural network. Size (batch_size, N_out)
        """
        """This condition is for training on train set as the validation set
            should not be trained, so we applied check condition(debug)
            debug : True -> validation set
            debug : False -> trainset
            The for loop will take samples from minibatch and train the network"""
            
        outputs = self.forward(inputs)
        dw1, db1, dw2, db2, dw3, db3 = self.backward(inputs, labels, outputs)
        self.weights, self.biases = optimizer.mbgd(self.weights, self.biases, dw1, db1, dw2, db2, dw3, db3, lr)
        creloss = loss.cross_entropy_loss(outputs,labels)
        accuracy = self.accuracy(outputs,labels)
         
        
        if debug:
            print('Training loss ',creloss)
            print('training accuracy: ',accuracy*100,'%')
          
         
        
        return creloss,accuracy, outputs.type(torch.FloatTensor)

    def predict(self, inputs):
        """Predicts output probability and index of most activating neuron

        This function is used to predict output given inputs. You can then use index in classes to show which class got activated. For example, if in case of MNIST fifth neuron has highest firing probability, then class[5] is the label of input.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 

        Returns:
            score (torch.tensor): max score for each class. Size (batch_size)
            idx (torch.tensor): index of most activating neuron. Size (batch_size)  
        """
        outputs = self.forward(inputs) # forward pass
        
        score , idx = torch.max(outputs,1)[0],torch.argmax(outputs,1)
        return score.type(torch.FloatTensor), idx.type(torch.LongTensor)

    def eval(self, inputs, labels, debug=False):
        """Evaluate performance of neural network on inputs with labels.

        This function is used to evaluate loss and accuracy of neural network on new examples. Unlike predict(), this function will not only predict but also calculate and return loss and accuracy w.r.t given inputs and labels.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 
            labels (torch.tensor): correct labels. Size (batch_size)
            debug (bool, optional): print loss and accuracy on every iteration. Defaults to False

        Returns:
            loss (float): average cross entropy loss
            accuracy (float): ratio of correctly to uncorrectly classified samples
            outputs (torch.tensor): predictions from neural network. Size (batch_size, N_out)
        """
        """We first take the average of loss and accuracy of every batch and 
            return"""
        temp_cre = 0
        temp_acc = 0
        outputs = 0
        m = inputs.shape[0]
        outputs = self.forward(inputs)
        for i in range(m):
            temp_cre += loss.cross_entropy_loss(outputs[i].reshape(1,-1),labels[i])
            temp_acc += self.accuracy(outputs[i].reshape(1,-1),labels[i])
         
            if debug:
                print('loss: ', temp_cre/(i+1))
                print('accuracy: ', temp_acc*100/(i+1))
        return temp_cre/m, temp_acc*100/m, outputs

    def accuracy(self, outputs, labels):
        """Accuracy of neural network for given outputs and labels.
        
        Calculates ratio of number of correct outputs to total number of examples.

        Args:
            outputs (torch.tensor): outputs predicted by neural network. Size (batch_size, N_out)
            labels (torch.tensor): correct labels. Size (batch_size)
        
        Returns:
            accuracy (float): accuracy score 
        """
        """The index with the max value in output tensor will be 
            considered as number prediction from the trained network
            then it is compared with the label and the correct numbers are 
            incremented, at last we average the corrrect prediction"""
        correct = 0
        m = outputs.shape[0]
        for i in range(m):
            temp = torch.argmax(outputs[i])
            if (temp.item() == labels[i].item()):
                correct+=1
        accuracy = correct/m
        return accuracy

    def forward(self, inputs):
        """Forward pass of neural network

        Calculates score for each class.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 

        Returns:
            outputs (torch.tensor): predictions from neural network. Size (batch_size, N_out)
        """
        """The linear outputs are first calculated at every layer then a 
            non linear funtion is applied to it and stored in cache for furter use"""
        self.cache['z1'] = self.weighted_sum(inputs,self.weights['w1'],self.biases['b1'])
        a1 = activation.sigmoid(self.cache['z1'])
        self.cache['z2'] = self.weighted_sum(a1,self.weights['w2'],self.biases['b2']) 
        a2 = activation.sigmoid(self.cache['z2'])
        self.cache['z3'] = self.weighted_sum(a2,self.weights['w3'],self.biases['b3'])
        
        outputs = activation.softmax(self.cache['z3']) 
        self.cache['a1'] = a1
        self.cache['a2'] = a2
        self.cache['a3'] = outputs
        
        return outputs

    def weighted_sum(self, X, w, b):
        """Weighted sum at neuron
        
        Args:
            X (torch.tensor): matrix of Size (K, L)
            w (torch.tensor): weight matrix of Size (J, L)
            b (torch.tensor): vector of Size (J)

        Returns:
            result (torch.tensor): w*X + b of Size (K, J)
        """
        mm = torch.mm(X,w.t())
        result = torch.add(mm,b)
        return result

    def backward(self, inputs, labels, outputs):
        """Backward pass of neural network
        
        Changes weights and biases of each layer to reduce loss
        
        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 
            labels (torch.tensor): correct labels. Size (batch_size)
            outputs (torch.tensor): outputs predicted by neural network. Size (batch_size, N_out)
        
        Returns:
            dw1 (torch.tensor): Gradient of loss w.r.t. w1. Size like w1
            db1 (torch.tensor): Gradient of loss w.r.t. b1. Size like b1
            dw2 (torch.tensor): Gradient of loss w.r.t. w2. Size like w2
            db2 (torch.tensor): Gradient of loss w.r.t. b2. Size like b2
            dw3 (torch.tensor): Gradient of loss w.r.t. w3. Size like w3
            db3 (torch.tensor): Gradient of loss w.r.t. b3. Size like b3
        """
        # Calculating derivative of loss w.r.t weighted sum
        """Calculating error to loss from outputs and labels"""
        dout = loss.delta_cross_entropy_softmax(outputs , labels)
        """Second layer error"""
        d2 = torch.mm(dout , self.weights['w3']) * activation.delta_sigmoid(self.cache['z2'])
        """First layer error"""
        d1 = torch.mm(d2 , self.weights['w2'] ) * activation.delta_sigmoid(self.cache['z1'])
        
        """Gradients for after backpropogation """
        dw1, db1, dw2, db2, dw3, db3 = self.calculate_grad(inputs, d1, d2, dout)# calculate all gradients
        return dw1, db1, dw2, db2, dw3, db3

    def calculate_grad(self, inputs, d1, d2, dout):
        """Calculates gradients for backpropagation
        
        This function is used to calculate gradients like loss w.r.t. weights and biases.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 
            dout (torch.tensor): error at output. Size like aout or a3 (or z3)
            d2 (torch.tensor): error at hidden layer 2. Size like a2 (or z2)
            d1 (torch.tensor): error at hidden layer 1. Size like a1 (or z1)

        Returns:
            dw1 (torch.tensor): Gradient of loss w.r.t. w1. Size like w1
            db1 (torch.tensor): Gradient of loss w.r.t. b1. Size like b1
            dw2 (torch.tensor): Gradient of loss w.r.t. w2. Size like w2
            db2 (torch.tensor): Gradient of loss w.r.t. b2. Size like b2
            dw3 (torch.tensor): Gradient of loss w.r.t. w3. Size like w3
            db3 (torch.tensor): Gradient of loss w.r.t. b3. Size like b3
        """
        """Updating weight gradients (dependent of previous layers)"""
        
        dw3 = torch.mm(dout.t(),self.cache['a2'])
        dw2 = torch.mm(d2.t() , self.cache['a1']) 
        dw1 = torch.mm(d1.t() , inputs)
        
        """Updating baises gradients (independent for previous layer)
            flatten method is used for 1D conversion"""
        db3 = torch.sum(dout ,0, True).flatten()
        db2 = torch.sum(d2 ,0, True).flatten() 
        db1 = torch.sum(d1 ,0, True).flatten() 
        return dw1, db1, dw2, db2, dw3, db3


if __name__ == "__main__":
    import activation, loss, optimizer
else:
    from nnet import activation, loss, optimizer

