
import torch

def sigmoid(z):
    """Calculates sigmoid values for tensors

    """
    result = 1.0 / (1 + torch.exp(-z))
    return result

""" The derivative of sigmoid can be written as exp(-z) * ( 1 / (1 + exp(-z)**2))
    on further simplifying (subraction of inner bracket value by 1 and taking sigmoid funtion common) 
    we get sigmoid(1 - sigmoid)"""
def delta_sigmoid(z):
    """Calculates derivative of sigmoid function

    """
    grad_sigmoid = sigmoid(z) * ( 1 - sigmoid(z))
    return grad_sigmoid

""" The Softmax outputs 0 or near zero value when overflow of exp occurs
    to tackle this we add a constant c to parameter and pass to funtion
    subracting max value from parameter leaves only the non-positive entries,
    thus ruling out overflow problem"""
def softmax(x):
    """Calculates stable softmax (minor difference from normal softmax) values for tensors

    """
    m = x.shape[0]
    z = x - torch.max(x)
    numerator = torch.exp(z)
    denominator = torch.sum(numerator,1)
    stable_softmax = numerator/denominator.reshape(m,1)
  
    return stable_softmax

if __name__ == "__main__":
    pass
