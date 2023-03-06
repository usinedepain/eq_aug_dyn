from opt_utils import ParameterTracker,cluster_epoch
from datasets import Rotate, SimpleShapes
from models import RotationNet

import torch
import os


import sys

bce = torch.nn.functional.binary_cross_entropy



                            
def imloss(pred,label):
    
    t,p = label

    return torch.nn.functional.binary_cross_entropy(pred[:,0,:],t.squeeze(1)) + torch.nn.functional.binary_cross_entropy(pred[:,1,:],p.squeeze(1))
    
    


if __name__ == '__main__':
    
    # parameters
    aug_nmbr = int(sys.argv[1])
    runs = int(sys.argv[2])
    
    if not os.path.isdir('results'):
        os.mkdir('results')
    
    for run in range(runs):
        name = os.path.join('results', 'res_rot_0'+str(run))
    
    
        lr = 1e-5
        loss = torch.nn.CrossEntropyLoss(reduction='mean')
        
        n=14  
        
        # define models
        print("Experiment "+ str(run) + " started")
        name = 'res_rot_0'+str(run)
        eqmodel = RotationNet(ns=[1,32,32,16,2],n=n, device = 'cuda', fully_connected=False)
        
            
        # start at equivariant point
        eqmodel.project_layers()
              
        tracker = ParameterTracker(eqmodel)
        # start augmented model at same point
        augmodel = eqmodel.copy()
        nonaugmodel = eqmodel.copy()
        
    
        shps = SimpleShapes(n,3000,directory = 'simpshape',device ='cuda')
        batch_size = 25
        loader = torch.utils.data.DataLoader(shps,batch_size=batch_size)
        
        rot = Rotate(n)
        
        config = {}
        config['eqmodel'] =  eqmodel
        config['augmodel'] = augmodel
        config['nonaugmodel'] = nonaugmodel 
        config['loader'] = loader
        config['non_loader'] = loader
        config['tracker'] =   tracker   
        config['lr'] = lr 
        config['name'] = name 
        config['augnumber'] = aug_nmbr   
        config['transformation'] = rot
        config['label_transformation'] =  rot
        config['loss'] = imloss
    
        if not os.path.isdir(name):
             os.mkdir(name)
        
        #overwrite results
        for tp in  ['eq','aug','nonaug']:
            os.remove(os.path.join(name,tp))
        
            
        #mx,mn = empirical_equivariance(model,loader,25)'
        for epochs in range(3):
            cluster_epoch(config)   

