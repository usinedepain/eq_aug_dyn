# Optimization Dynamics of Equivariant and Augmented Multilayer Perceptrons

This repository contains code used for the experiments in the paper ... . 

## Description of files in the repository
* models.py contain the code for building the models used in the experiment. 
It is built around an abstract class ProjectNet, which is instantiated for the three different group. 
The ProjectNet stores a number of layers and a method for projecting all layers (and gradients) onto the space E.
* datasets.py contains code for handling (and generating) data as used in the experiments. 
* opt_utils contains boiler-plate code for training the MLP:s using the different approaches of the paper. 
* eval_script.py is a script for evaluating results as stored by the training files.
* train_permutation.py, train_translation.py, train_rotation.py are training scripts for repeating the experiments in the paper
* simpshape is the dataset used in the rotation experiment.
* stochblock is the dataset used for the permutation experiment. 

The datasets has a total size of about 200 MB.


## Running experiments
To run an experiment, run one of the training scripts train_permutation.py, train_translation.py or train_rotation.py with arguments augnumber nbr_experiments. 
Augnumber decides how many passes should be made over the data for the augmented run (in the experiments in the paper, these were set to 25). I.e., to run the permutation experiment as in the paper, run

```

    python train_permutation.py 25 30
    
```
 
Then, run evalscript.py with one of the key values 'perm', 'trans' or 'rot' as first parameter and nbr_exp as second parameter to produce a figure like the one in the paper (it will be less tidy
have the same general appearance). To evaluate the permutation experiment, run

```

   python evalscript.py perm 30
   
```

### Disclaimers
The code has been modified slightly for release. In particular, we ran our code on a cluster to perform the experiments in the paper, which required some minor changes. We do not expect the code to behave differently (apart from the random initializations), but guarantees cannot be given.

Note that in particular the translation and rotation experiments will take quite some time, in particular when repeating them 30 times. The code has not been optimized in any way, and our experiments took around 75 errors in total. For exploratory tests, we recommend using the permutation examples.

The experiments will default to be performed on the data downloaded with the repository, which is the data we used. The datasets classes are made ready to generate new sets of data if you wish to do so. See further documentation in datasets.py.



