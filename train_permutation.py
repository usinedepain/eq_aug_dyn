from datasets import adMat, identity
import torch
import os
from models import PermutationAdjNet
import sys

from opt_utils import ParameterTracker, cluster_epoch

import time as t

"""
    Training 

"""

if __name__ == '__main__':
    
    # parameters
    aug_nmbr = int(sys.argv[1])
    runs = int(sys.argv[2])
    
    if not os.path.isdir('results'):
        os.mkdir('results')
    
    for run in range(runs):
        name = os.path.join('results', 'res_perm_0'+str(run))
        
        lr = 1e-5
        loss = torch.nn.BCELoss(reduction='mean')
        
        if not os.path.exists(os.path.join(name)):
            os.mkdir(os.path.join(name))
            
            
           
        print("Experiment "+ str(run) + " started")
       
        # define models
        eqmodel = PermutationAdjNet(ns=[1,32,],n=10, device = 'cuda', fully_connected=True,ms = [64,32])
           
        eqmodel.to('cuda')
            
            
            
        # start at equivariant point
        eqmodel.project_layers()
        for layer in eqmodel.layers:
            layer.data = layer.data
                
        tracker = ParameterTracker(eqmodel)
        # start (non)-augmented model at same point
        augmodel = eqmodel.copy()
        nonaugmodel = eqmodel.copy()
            
        
        # define dataloaders for augmented and non-augmented training
        # 
        permer = adMat(directory = 'stoch_block', device = 'cuda',aug=True, nbr_samples=1000,n=5)
        nonpermer = adMat(directory = 'stoch_block', device='cuda',aug=False, nbr_samples=1000, n=5)    
            
        loader = torch.utils.data.DataLoader(permer,batch_size=200)
        non_loader = torch.utils.data.DataLoader(nonpermer,batch_size=200)
        
        config = {}
        config['eqmodel'] =  eqmodel
        config['augmodel'] = augmodel
        config['nonaugmodel'] = nonaugmodel 
        config['loader'] = loader
        config['non_loader'] = non_loader
        config['tracker'] =  tracker   
        config['lr'] = lr 
        config['name'] = name 
        config['augnumber'] = aug_nmbr   
        config['transformation'] = permer.permuteAdj
        config['label_transformation'] =  identity
        config['loss'] = loss
        
        
        
        if not os.path.isdir(name):
            os.mkdir(name)
            
            
        #overwrite results
        for tp in  ['eq','aug','nonaug']:
            fln = os.path.join(name,tp)
            if os.path.exists(fln):
                os.remove(fln)
        
        for epochs in range(50):
            cluster_epoch(config)    

