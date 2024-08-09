# Optimization Dynamics of Equivariant and Augmented Neural Networks

This repository contains code used for the experiments in the two versions of the [paper](https://arxiv.org/abs/2303.13458)

```
@article{nordenfors2024optEqAug,
  title={Optimization Dynamics of Equivariant and Augmented Neural Networks},
  author={Nordenfors, Oskar and Ohlsson, Fredrik and Flinth, Axel},
  journal={arXiv:2303.13458},
  year={2024}
}
```

It is mainly released for transparency and reproducibility purposes. Should you find this code useful for your research, please cite the above paper.

## Description of files in the repository
* models.py and networks.py contain the code for building the models used in the experiment. 
It is built around an abstract class ProjectNet, which is instantiated for the three different group. 
The ProjectNet stores a number of layers and a method for projecting all layers (and gradients) onto the space E.
* datasets.py contains code for handling (and generating) data as used in the experiments. 
* opt_utils.py, trainer_local.py, tester_local.py and mymse.py contains boiler-plate code for training the MLP:s using the different approaches of the paper. 
* eval_script.py  and plot_experiment_data.py are script for evaluating and plotting results as stored by the training files.
* eqv_vs_aug_dyn_experiment_local.py, train_permutation.py, train_translation.py, train_rotation.py, train_different_groups.py are training scripts for repeating the experiments in the two versions of the paper
* simpshape is the dataset used in the rotation experiment in the first paper.
* stochblock is the dataset used for the permutation experiment in the first paper. 

### Required and optional libraries
The package requires the [pytorch](https://pytorch.org/) and [numpy](https://numpy.org/), [math](https://docs.python.org/3/library/math.html), [os](https://docs.python.org/3/library/os.html) and [tqdm](https://tqdm.github.io/)  packages .  [matplotlib](https://matplotlib.org/) is optional: it is only needed for the plots in eval_script.py. Also optional is [cv2](https://pypi.org/project/opencv-python/). It is used also used for the simpshape dataset in datasets.py, but only when generating new datasets. It does not need to be installed if the downloaded dataset is used.

## Running experiments

### Main experiment

To run the experiment from the paper simply run 
```
  python eqv_vs_aug_dyn_experiment_local.py.
```
This may take a while depending on the computer used to run it. If one wishes to run a scaled-down version of the experiment one can go into the code and change the number of experiments and/or the number of training epochs (The corresponding parameters are situated near the top of the code). Note that the number of experiments should be divisible by 3, because there are 3 separate trials which are compared.

### Experiments from the appendix and first version of the paper
To run an experiment, run one of the training scripts train_permutation.py, train_translation.py or train_rotation.py with arguments augnumber nbr_experiments. 
Augnumber decides how many passes should be made over the data for the augmented run (in the experiments in the paper, these were set to 25). I.e., to run the permutation experiment as in the paper, run

```
    python train_permutation.py 25 30
```
 
Then, run evalscript.py with one of the key values 'perm', 'trans' or 'rot' as first parameter and nbr_exp as second parameter to produce a figure like the one in the paper (it will be less tidy but
have the same general appearance). To evaluate the permutation experiment, run

```
   python evalscript.py perm 30
```

### The multiple group experiment
To run the experiment in the appendix, use the file train_different_groups.py just as above. I.e., to repeat them just as in the paper, run

```
    python train_different_groups.py 25 30
```

Then run evalscript.py with key value 'all_t' as first parameter, e.g.

```
   python evalscript.py all_t 30
```

The plot windows may appear on top of each other -- move them to inspect them all at once.

### Disclaimers
The experiments in the paper were performed on a cluster. Therefore, some aspects of the code, such as the training scripts, differs very slightly from the code released here. There should be no difference in performance.

Note that in particular the translation and rotation experiments will take quite some time, in particular when repeating them 30 times. The code has not been optimized in any way, and our experiments took around 135 GPU hours in total. For exploratory tests, we recommend using the permutation examples.

The experiments will default to be performed on the data downloaded with the repository, which is the data we used. The datasets classes are made ready to generate new sets of data if you wish to do so. See further documentation in datasets.py.



