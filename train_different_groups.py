from datasets import adMat, identity, SimpleShapes, Rotate, Shift, get_MNIST_loader, Xshift, RotoShift
import torch
import os
from models import PermutationAdjNet, RotationNet, TranslationNet, OneDTranslationNet, RotoTransNet
import sys

from opt_utils import ParameterTracker, cluster_epoch, layer_equivariance


bce = torch.nn.functional.binary_cross_entropy

"""
    Train four different models on MNIST with different symmetry groups
"""




if __name__ == '__main__':
    
    # parameters
    experiment_types = ['rot_class','trans','trans_one','rot_trans'] # the four groups
    aug_nmbr = int(sys.argv[1])
    runs = int(sys.argv[2])
    
    if not os.path.isdir('results'):
        os.mkdir('results')
    
    data_root = 'mnist'
      
    
    for experiment_type in experiment_types:
        # make directory for saving results
        result_dir = 'results'
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)
        
        for run in range(runs):
            
            # Define meta parameters
            config = {}
            
            # common ones for all experiments
            config['lr'] = 5e-5
            config['loss'] = torch.nn.CrossEntropyLoss(reduction='mean')
            config['loader'] = get_MNIST_loader(25,14,root=data_root)
            config['non_loader'] = get_MNIST_loader(25,14,root=data_root)
            config['label_transformation'] = identity
            config['augnumber'] = aug_nmbr
            name = os.path.join(result_dir, 'res_'+experiment_type +'_0' + str(run))
            config['name'] = name
            n=14
                              
            # specific parameters for the groups
            
            if experiment_type == "rot_class":
                config['transformation'] = Rotate(n,True)
                eqmodel = RotationNet(ns = [1,32,32], n=14, device = 'cuda', fully_connected = True, ms = [32])
            elif experiment_type == "trans":
                config['transformation'] = Shift(n)
                eqmodel =  TranslationNet(ns=[1,32,32],n=14, device = 'cuda', fully_connected=True, ms = [32])
            elif experiment_type == "trans_one":
                config['transformation'] = Xshift(n)
                eqmodel = OneDTranslationNet(ns = [1,32,32], n=14, device = 'cuda', fully_connected = True, ms = [32])
            elif experiment_type == 'rot_trans':
                config['transformation'] = RotoShift(n)
                eqmodel = RotoTransNet(ns = [1,32,32], n=14, device = 'cuda', fully_connected = True, ms = [32])
               
            eqmodel.to('cuda')
                
                  
            # start at equivariant point
            eqmodel.project_layers()

            #track changes
            config['tracker'] = ParameterTracker(eqmodel)
            
            # start (non)-augmented model at same point
            augmodel = eqmodel.copy()
            nonaugmodel = eqmodel.copy()
                
            config['eqmodel'] =  eqmodel
            config['augmodel'] = augmodel
            config['nonaugmodel'] = nonaugmodel
           
            
            
            print("Experiment "+ experiment_type +", run "+ str(run) + " started")
            if not os.path.isdir(name):
                os.mkdir(name)
                
                
            #overwrite results
            for tp in  ['eq','aug','nonaug']:
                fln = os.path.join(name,tp)
                if os.path.exists(fln):
                    os.remove(fln)
            
            for epochs in range(50):
                cluster_epoch(config)  