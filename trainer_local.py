import numpy as np
import torch
from tqdm import tqdm

def train(model, loader, optimizer, criterion, device):

    running_loss = np.zeros(len(loader))
    running_acc = 0.0
    running_grad_norm = np.zeros(len(loader))
    running_distance = np.zeros((4,len(loader))) # 4 is the number of layers

    i=0
    model.train()
    for data in tqdm(loader,desc='Training progress', leave=False):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # If model is equivariant do nothing
            if model.is_equivariant():
                pass
            # Otherwise, calculate the distance to the symmetric subspace
            else:
                a,b,c,d = model.calculate_weight_tensor()
                x,y,z,w = model.projection()
                running_distance[0,i] = torch.linalg.norm(a-x)
                running_distance[1,i] = torch.linalg.norm(b-y)
                running_distance[2,i] = torch.linalg.norm(c-z)
                running_distance[3,i] = torch.linalg.norm(d-w)
            acc = outputs.argmax(1,keepdim=True).eq(labels.view_as(outputs.argmax(1,keepdim=True))).sum().float() / labels.shape[0]
            gradient_norm=0
            for p in model.parameters():
                gradient_norm += p.grad.detach().data.norm(2).item() ** 2
            gradient_norm = gradient_norm ** 0.5
            running_grad_norm[i] = gradient_norm
        running_loss[i] = loss.item()
        running_acc += acc.item()
        i+=1
    
    running_acc = running_acc/len(loader)
    return running_loss, running_acc, running_grad_norm, running_distance