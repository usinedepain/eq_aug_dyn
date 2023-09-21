# Optimization Dynamics of Equivariant and Augmented Multilayer Perceptrons

![a decorative plot with three curves and a stable point](https://github.com/usinedepain/eq_aug_dyn/master/figure/dynamics_plot.png)

This repository contains code used for the experiments in the [paper](https://arxiv.org/abs/2303.13458)

```
@article{flinth2023optEqAug,
  title={Optimization Dynamics of Equivariant and Augmented Neural Networks},
  author={Flinth, Axel and Ohlsson, Fredrik},
  journal={arXiv:2303.13458},
  year={2023}
}
```

It is mainly released for transparency and reproducibility purposes. Should you find this code useful for your research, please cite the above paper.

## Description of files in the repository
* models.py contain the code for building the models used in the experiment. 
It is built around an abstract class ProjectNet, which is instantiated for the three different group. 
The ProjectNet stores a number of layers and a method for projecting all layers (and gradients) onto the space E.
* datasets.py contains code for handling (and generating) data as used in the experiments. 
* opt_utils contains boiler-plate code for training the MLP:s using the different approaches of the paper. 
* eval_script.py is a script for evaluating results as stored by the training files.
* train_permutation.py, train_translation.py, train_rotation.py, train_different_groups.py are training scripts for repeating the experiments in the paper
* simpshape is the dataset used in the rotation experiment.
* stochblock is the dataset used for the permutation experiment. 

### Required and optional libraries
The package requires the [pytorch](https://pytorch.org/) and [numpy](https://numpy.org/) packages.  [matplotlib](https://matplotlib.org/) is optional: it is only needed for the plots in eval_script.py. Also optional is [cv2](https://pypi.org/project/opencv-python/). It is used also used for the simpshape dataset in datasets.py, but only when generating new datasets. It does not need to be installed if the downloaded dataset is used.

## Running experiments
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



