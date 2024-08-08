import torch.nn.functional as F

def mse(output, label):
    one_hot_label = F.one_hot(label,10) # Cast labels as one-hot vectors
    return F.mse_loss(output, one_hot_label/1.0) # Return MSE loss