from context import nnet
from nnet import loss, activation
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from nnet import model
import numpy.linalg.linalg as li
import unittest
import torch
import math
import numpy as np

from nnet import loss
N_in = 28 * 28 # Input size
N_h1 = 256 # Hidden Layer 1 size
N_h2 = 256 # Hidden Layer 2 size
N_out = 10 # Output size
lr = 0.01
device = "cuda:0" if torch.cuda.is_available() else "cpu"
net = model.FullyConnected(N_in, N_h1, N_h2, N_out, device=device)

def costFunction(X, y):
        """Compute cost for given X,y, use weights already stored in class."""
        yHat = net.forward(X)
        J = loss.cross_entropy_loss(yHat,y)
        return J


def costFunctionPrime(X, y):
        """Compute derivative with respect to W1,W2,W3,B1,B2,B3 for a given X and y:"""
        yHat = net.forward(X)
        
        dw1, db1, dw2, db2, dw3, db3 = net.backward(X,y,yHat)
        
        return  dw1, db1, dw2, db2, dw3, db3
def getParams():
        #Get W1,W2,W3,B1,B2,B3 unrolled into vector:
        params = torch.cat(( net.weights['w1'].flatten(), net.weights['w2'].flatten(), net.weights['w3'].flatten() ,net.biases['b1'].flatten() ,net.biases['b2'].flatten(),net.biases['b3'].flatten() ))
        return params

def setParams( params):
        """Set W1,W2,W3,B1,B2,B3 using single paramater vector."""
        W1_start = 0
        
        W1_end = net.N_h1 * net.N_in
        net.weights['w1'] = torch.reshape(params[W1_start:W1_end], (net.N_h1 , net.N_in))
       
        W2_end = W1_end + net.N_h2 * net.N_h1
        net.weights['w2'] = torch.reshape(params[W1_end:W2_end], (net.N_h2 , net.N_h1))
       
        W3_end = W2_end + net.N_out * net.N_h2
        net.weights['w3'] = torch.reshape(params[W2_end:W3_end], (net.N_out , net.N_h2))
       
        b1_end = W3_end +net.N_h1
        net.biases['b1'] = torch.reshape(params[W3_end:b1_end], (net.N_h1,1))
      
        b2_end = b1_end +net.N_h2
        net.biases['b2'] = torch.reshape(params[b1_end:b2_end], (net.N_h2,1))
        
        b3_end = b2_end +net.N_out
        net.biases['b3'] = torch.reshape(params[b2_end:b3_end], (net.N_out,1))
        
        net.biases['b1'] = net.biases['b1'].flatten()
        net.biases['b2'] = net.biases['b2'].flatten()
        net.biases['b3'] = net.biases['b3'].flatten()
       
def computeNumericalGradient(X, y):
        """Will compute numeric gradient by using finite diffrence method"""
        paramsInitial = getParams()
        numgrad = torch.zeros(paramsInitial.shape)
        perturb = torch.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            setParams(paramsInitial + perturb)
            loss2 = costFunction(X, y)
            
            setParams(paramsInitial - perturb)
            loss1 = costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        setParams(paramsInitial)

        return numgrad 

def computeGradients(X, y):
        dw1, db1, dw2, db2, dw3, db3 = costFunctionPrime(X, y)
        
        return torch.cat(( dw1.flatten(), dw2.flatten(), dw3.flatten(), db1.flatten(), db2.flatten(), db3.flatten() ))

class TestLossModule(unittest.TestCase):
    # Extra TODO: Write more rigorous tests

    def test_cross_entropy(self):
        # settings
        batch_size = 11
        N_out = 12

        # tensors
        outputs = torch.rand((batch_size, N_out), dtype=torch.float)
        labels = torch.randint(high=N_out, size=(batch_size,), dtype=torch.long)
        #test case for nan outputs
        temp = []
        for i in range(len(outputs)):
            temp.append((outputs[i] >= 1e-320).all())
        assert any(temp)
        
        creloss = loss.cross_entropy_loss(activation.softmax(outputs), labels)
        assert type(creloss) == float
        # write more robust and rigourous test cases here
        nll = torch.nn.functional.cross_entropy(outputs, labels)
        self.assertAlmostEqual(creloss, nll.item(), places=6)

    def test_delta_cross_entropy_loss(self):
        # settings
        batch_size = 11
        N_out = 17
        precision = 0.000001

        # tensors
        outputs = torch.rand((batch_size, N_out), dtype=torch.float, requires_grad=True)
        labels = torch.randint(high=N_out, size=(batch_size,), dtype=torch.long)

        # calculate gradients from scratch
        grads_creloss = loss.delta_cross_entropy_softmax(activation.softmax(outputs), labels)
        
        # calculate gradients with autograd
        nll = torch.nn.functional.cross_entropy(outputs, labels)
        nll.backward()

        assert isinstance(grads_creloss, torch.FloatTensor)
        assert grads_creloss.size() == torch.Size([batch_size, N_out])
        print(torch.le(torch.abs(grads_creloss - outputs.grad), precision).all())
        self.assertTrue(torch.le(torch.abs(grads_creloss - outputs.grad), precision).all())
        # write more robust test cases here
        # you should write gradient checking code here
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', 
                                                          download=True, 
                                                          train=False,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                              transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                          ])),batch_size=1, shuffle=True)
    
        #for single iteration
        temp = next(iter(test_loader))
        #Computing numerical gradient and backprop gradient
        #Indexes are used to retrive single row data and target
        numgrad  = computeNumericalGradient(temp[0][:1].reshape(1,784),temp[1][:1].reshape(1))
        actual_grad = computeGradients(temp[0][:1].reshape(1,784),temp[1][:1].reshape(1))
        #Checking condition
        assert li.norm(actual_grad - numgrad)/li.norm(actual_grad + numgrad) <0.5
        
        
        

if __name__ == '__main__':
    unittest.main()