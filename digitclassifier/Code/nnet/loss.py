import torch

"""The cross entropy loss was given for binary class outputs
    but for multiclass outputs we used negetive log likelihood,
    the labels as one-hot-encoded thus the labels with one will 
    support in error calculation"""
def cross_entropy_loss(outputs, labels):
    """Calculates cross entropy loss given outputs and actual labels

    """ 
    """One hot ending for labels and  reshaping"""
    
    m = outputs.shape[0]
    n = outputs.shape[1]
    temp_labels = torch.zeros(m,n)
    for i in range(m):
        temp_labels[i][labels[i]] = 1
    creloss = -(1/m)*torch.sum( temp_labels * torch.log(outputs)  )


    return creloss.item()   # should return float not tensor

    
"""We calculated the derivative of negetive likehood w.r.t to softmax and multiplied it
   with  derivative of softmax w.r.t to output from last previous activation funtion"""  
    
def delta_cross_entropy_softmax(outputs, labels):
    """Calculates derivative of cross entropy loss (C) w.r.t. weighted sum of inputs (Z). 
    
    """
    
    '''processing labels(One-hot encoding) '''
    m = outputs.shape[0]
    n = outputs.shape[1]
    
    temp_labels = torch.zeros(m,n)
    for i in range(m):
        temp_labels[i][labels[i]] = 1
    
    avg_grads = outputs - temp_labels
    
    return avg_grads*(1/m)

if __name__ == "__main__":
    pass
