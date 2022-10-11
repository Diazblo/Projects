from context import nnet
from nnet import activation

import unittest
import torch
import math
import random

class TestActivationModule(unittest.TestCase):
    # Extra TODO: Write more rigorous tests

    def test_sigmoid(self):
        # This is an example test case. We encourage you to write more such test cases.
        # You can test your code unit-wise (functions, classes, etc).
        x = torch.FloatTensor([[-140, -0.2, -0.6, 0, 0.1, 0.5, 2, 50], [-1, -20, -0.8, 10, 1, 0.5, 2.771, 41]])
        # y = torch.FloatTensor([[4.53979e-05, 0.45016, 0.35434, 0.5, 0.52498, 0.62246, 0.88079, 0.9999], 
        #                         [4.53979e-05, 0.45016, 0.35434, 0.5, 0.52498, 0.62246, 0.88079, 0.9999]])
        precision = 0.000001
        self.assertTrue(torch.le(torch.abs(activation.sigmoid(x) - x.sigmoid()), precision).all())
        
        """This will check the condition for range in sigmoid """
        x = torch.FloatTensor([[10, -1.2, 0.9, 0, -0.3, 0.4, 7, 44], [-89, -0.6, 0.1, 0, 0.5, 0.1, 5, 90]])
        y = activation.sigmoid(x)
        assert torch.min(y) >= 0 and torch.max(y) <=1
	
        
        
    def test_delta_sigmoid(self):
        batch_size = 6
        N_hn = 512
        precision = 0.000001 

        x = torch.rand((batch_size, N_hn), dtype=torch.float, requires_grad=True)
        grads = activation.delta_sigmoid(x)
        
        # calculate gradients with torch
        x.sigmoid().backward(torch.ones_like(x))
        
        assert isinstance(grads, torch.FloatTensor)
        assert grads.size() == torch.Size([batch_size, N_hn])
        self.assertTrue(torch.le(torch.abs(grads - x.grad), precision).all())

    def test_softmax(self):
        batch_size = 7
        N_out = 10
        precision = 0.000001
        
        x = torch.rand((batch_size, N_out), dtype=torch.float)
        outputs = activation.softmax(x)
        
    
        
        assert isinstance(outputs, torch.FloatTensor)
        assert outputs.size() == torch.Size([batch_size, N_out])
        self.assertTrue(torch.le(torch.abs(outputs - x.softmax(1)), precision).all())  


if __name__ == '__main__':
    unittest.main()