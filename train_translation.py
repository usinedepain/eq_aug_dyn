from opt_utils import restrictedSGD_epoch, augmentedSGD_epoch, layer_equivariance,empirical_equivariance, identity, ParameterTracker,cluster_epoch
from datasets import Shift, get_MNIST_loader, Xshift
from models import TranslationNet, OneDTranslationNet

import torch
import os

import sys

import time as t




if __name__ == '__main__':
        
    aug_nmbr = int(sys.argv[1])
    runs = int(sys.argv[2])
    
    if not os.path.isdir('results'):
        os.mkdir('results')
    
    for run in range(runs):
        name = os.path.join('results', 'res_trans_0'+str(run))
    # parameters    
        lr =  1e-5
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

    
        print("Experiment "+ str(run) + " started")
   
        eqmodel = OneDTranslationNet(ns=[1,32,32],n=14, device = 'cuda', fully_connected=True, ms = [32])

       
        eqmodel.to('cuda')
        
     
        
        
        # start at equivariant point
        
        eqmodel.project_layers()
        for (k,layer) in enumerate(eqmodel.layers):
            layer.data = layer.data
          
            tracker = ParameterTracker(eqmodel)
            # start augmented model at same point
            augmodel = eqmodel.copy()
            nonaugmodel = eqmodel.copy()
            loader = get_MNIST_loader(25,14, root ='mnist')
        
        sh = Xshift(14)

        config = {}
        config['eqmodel'] =  eqmodel
        config['augmodel'] = augmodel
        config['nonaugmodel'] = nonaugmodel 
        config['loader'] = loader
        config['non_loader'] = loader
        config['test_loader'] = loader 
        config['tracker'] =   tracker   
        config['lr'] = lr 
        config['name'] = name 
        config['augnumber'] = aug_nmbr   
        config['transformation'] = sh
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

 




