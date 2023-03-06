from datasets import identity
import torch
import numpy as np
import os

"""
    Functions for handling the training of the models
"""

def restrictedSGD_epoch(model,lr,loss,loader):
    
    """
    go over the dataset once and update an equivariant model.    
    
        model - the model to be trained
        lr - learning rate
        loss - the loss function
        loader - Data loader
    """

    output = 0.0
    
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    
    # K is the number of mini-batches
    # For proper normalization of the gradients.
    K = loader.dataset.__len__()/loader.batch_size
    
    total_error = 0.0
    for k, (x,label) in enumerate(loader): # iterate over dataset
            x=x.to(model.device)
            try:
                label=label.to(model.device)
            except(AttributeError):
                # For the rotation case, we need to handle two labels.
                # This was the easiest solution.
                for k,lab in enumerate(label):
                    label[k] = lab.to(model.device)
            pred = model(x)
            output = loss(pred,label)/K
            
            total_error +=  output.cpu().detach().numpy()
          
            # when we call backward, pytorch stores the gradient, but does not update the model yet
          
            output.backward()
            
    # before updating the parameters, project the gradients onto E
    model.project_grads()
    
    # take a gradient step
    optimizer.step()
    optimizer.zero_grad()
    return total_error



def augmentedSGD_epoch(model,lr,loss,loader,L,transformation,label_transformation=identity):
    
    """
       Perform one epoch of  non-equivariant MLP training
   
           model - the model to be trained
           lr - learning rate
           loss - the loss function
           loader - Data loader
           L - the number of passes. If no augmentation is done, can set L=1
           transformation - transformation function fot the input .
           label_transformation - transformation functions for the labels
        
        To emulate NOMINAL training, set
            L=1, transformation = identity and label_transformation = identity
    """

        
    output = 0.0
    
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    # K is the number of mini-batches
    # For proper normalization of the gradients.
    K= loader.dataset.__len__()/loader.batch_size
    total_error = 0.0
    
    for l in range(L):
        for x,label in loader:
            x = x.to(model.device)
            
            x = transformation(x)
            
            try:
                label=label.to(model.device)
                label = label_transformation(label)
            except(AttributeError):
                # For the rotation case, we need to handle two labels.
                # This was the easiest solution
                for k,lab in enumerate(label):
                    lab = lab.to(model.device)
                    label[k] = label_transformation(lab)

            pred = model(x)
            output = loss(pred,label)/(K*L)
            
            total_error +=  output.cpu().detach().numpy()
        

            # when we call backward, pytorch stores the gradient, but does not update the model yet
          
            output.backward()
        
     
        
   
    # take a gradient step
    optimizer.step()
    optimizer.zero_grad()
    return total_error

            
def empirical_equivariance(model,loader,M, transform, label_transformation = identity):
    maxerr = torch.tensor(0.0)
    meanerr =torch.tensor(0.0)
    with torch.no_grad():
        for x,label in loader:
            x=x.to(model.device)
            try:
                label=label.to(model.device)
            except(AttributeError):
                for lab in label:
                    lab = lab.to(model.device)
            pred0 = model(x)
            for l in range(M):
                x = transform(x)
                for lab in label:
                    lab = label_transformation(lab)
                #if x.dim()==3:
                #    x = x.unsqueeze(1)
                pred = model(x)
                maxerr = torch.max(maxerr,torch.abs(pred-pred0).max())
                meanerr = torch.abs(pred-pred0).sum()
        meanerr = meanerr/loader.dataset.__len__()
         
    return maxerr.cpu().detach().numpy(), meanerr.cpu().detach().numpy()
    
def layer_equivariance(model):
    with torch.no_grad():
        nbr_full_layers = len(model.layers)-1
        errs = np.zeros(nbr_full_layers + 1)
        for k in range(nbr_full_layers):
            L = model.layers[k].data.reshape(model.ns[k+1],model.ns[k],model.n**2,model.n**2)
            Lp = model.project(L)
            
            errs[k] = ((L-Lp).cpu().detach().numpy()**2).sum()
        
        L = model.layers[-1].data
        Lp = model.project_light(L)
        
        errs[-1] = ((L-Lp).cpu().detach().numpy()**2).sum()
    return errs


"""
A class for keeping track of changes in parameters.

Saves an inital state, and can measure change from that at any time
"""
class ParameterTracker(torch.nn.Module):
    
    def __init__(self,model):
        self.inpoints = self.extract_data(model)
            
    def change(self,model):
        points = self.extract_data(model)
        
        dists = []
        
        for k in range(len(points)):
            
            dists.append(np.sqrt(((points[k]-self.inpoints[k])**2).sum()))
            
        return dists
        
        
    @staticmethod
    def extract_data(model):
        inpoints = []
        
        for layer in model.layers:
            inpoints.append(layer.cpu().detach().numpy())
        
        return inpoints
    
""" 
    boilerplate code for one epoch of training all three models
"""

def cluster_epoch(config):
    
    """
    
    call the model with following fields in config dictionary:
        
        the three models:
            'eqmodel'
            'augmodel'
            'nonaugmodel'
        loaders for non-augmented and augmented training
            'loader'
            'non_loader'
        An ParameterTracker to track the evolution of the models
            'tracker'
        The learning rate
            'lr'
        The number of passes to be made over the data for the augmented experiment
            'augnumber'
        Transformation routines for the inputs and the labels
            'transformation'
            'label_transformation'
        A loss function
            'loss'
    
    """
    
    eqmodel = config['eqmodel']
    augmodel = config['augmodel']
    nonaugmodel = config['nonaugmodel']
    
    
    loader = config['loader']
    non_loader = config['non_loader']
    tracker = config['tracker']
    lr = config['lr']
    name = config['name']
    augnumber = config['augnumber']
    transformation = config['transformation']
    label_transformation = config['label_transformation']
        
    loss = config['loss']
    
    eqmodel.train()
    augmodel.train()
    nonaugmodel.train()
    
    # take a step for all the models
    eqloss = restrictedSGD_epoch(eqmodel,lr,loss,loader)
    augloss = augmentedSGD_epoch(augmodel,lr,loss,loader,augnumber,transformation,label_transformation)
    nonaugloss = augmentedSGD_epoch(nonaugmodel,lr,loss,non_loader,1,identity,identity)

    eqmodel.eval()
    augmodel.eval()
    nonaugmodel.eval()    
    
    
    #measure the invariance of the layers
    eqlay = layer_equivariance(eqmodel)
    auglay = layer_equivariance(augmodel)
    nonauglay = layer_equivariance(nonaugmodel)

    # and the total change
    eqshunt = tracker.change(eqmodel)
    augshunt = tracker.change(augmodel)
    nonaugshunt = tracker.change(augmodel)
   

    # save the results in text-files to be processed by the eval scripts.
    f = open(os.path.join(name,'eq'), "a")
    f.write('loss :' + str(eqloss)+ ' lay: ' + str(eqlay) + ' shunt: '+ str(eqshunt) + '\n' )
    f.close() 
    
    f = open(os.path.join(name,'aug'), "a")
    f.write('loss :' + str(augloss)+' lay: ' + str(auglay) +  ' shunt: ' + str(augshunt) + '\n')
    f.close()

    f = open(os.path.join(name,'nonaug'), "a")
    f.write('loss :' + str(nonaugloss)+' lay: ' + str(nonauglay) + ' shunt: ' + str(nonaugshunt) + '\n')
    f.close() 

