import os
import numpy as np
import matplotlib.pyplot as plt

########################################

# THESE PARAMETERS HAVE TO MATCH THE ONES USED WHEN RUNNING eqv_vs_aug_dyn_experiment_local !
NO_EXPERIMENTS = 90 # total number of experiments
AUG_EPOCHS = 10 # number of augmented epochs

########################################


EXPERIMENTS_IND1 = int(NO_EXPERIMENTS/3)
EXPERIMENTS_IND2 = int(EXPERIMENTS_IND1*2)
AUG_BUFFER = AUG_EPOCHS + 1 # adds an extra dimension to allow plotting from 0.

# Load the data

PATH = os.getcwd()
LOAD_ROOT = os.path.join(PATH,r'EqvAugDyn/Local_Testing/')

def load_distance(name,start):
    ARRAY = np.zeros([EXPERIMENTS_IND1,AUG_BUFFER,4,2400])
    for i in range(EXPERIMENTS_IND1):
        ARRAY[i,1:,:,:] = np.load(LOAD_ROOT+'aug_distance_'+name+str(start+i)+'.npy')
    return ARRAY

DIST_CROSS = load_distance('cross',0)
DIST_SKEW = load_distance('skew',EXPERIMENTS_IND1)
DIST_NO = load_distance('cross_non_orthogonal',EXPERIMENTS_IND2)


# Plot the data
fig, ax = plt.subplots(1,2)

"========================================="

ax[0].plot(np.sqrt(np.sum(DIST_CROSS**2,-2)).mean(-1).T,color='#D81B60',linewidth=0.1)
mean_dist_cross, =ax[0].plot(np.median(np.sqrt(np.sum(DIST_CROSS**2,-2)).mean(-1),0).T,color='#D81B60')

ax[0].plot(np.sqrt(np.sum(DIST_SKEW**2,-2)).mean(-1).T,color='#1E88E5',linewidth=0.1)
mean_dist_skew, =ax[0].plot(np.median(np.sqrt(np.sum(DIST_SKEW**2,-2)).mean(-1),0).T,color='#1E88E5')

ax[0].plot(np.sqrt(np.sum(DIST_NO**2,-2)).mean(-1).T,color='#FFC107',linewidth=0.1)
mean_dist_no, =ax[0].plot(np.median(np.sqrt(np.sum(DIST_NO**2,-2)).mean(-1),0).T,color='#FFC107')

ax[0].legend([mean_dist_cross,mean_dist_skew,mean_dist_no],['Cross', 'Skew','Non-Unitary'])
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Distance from $ \\mathcal{E} $')
ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

"==========================================="

ax[1].plot(np.sqrt((DIST_CROSS[:,1,:,:]**2).sum(-2)).T,color='#D81B60',linewidth=0.1)
mean_dist_cross, =ax[1].plot(np.median(np.sqrt((DIST_CROSS[:,1,:,:]**2).sum(-2)),0).T,color='#D81B60')

ax[1].plot(np.sqrt((DIST_SKEW[:,1,:,:]**2).sum(-2)).T,color='#1E88E5',linewidth=0.1)
mean_dist_skew, =ax[1].plot(np.median(np.sqrt((DIST_SKEW[:,1,:,:]**2).sum(-2)),0).T,color='#1E88E5')

ax[1].plot(np.sqrt((DIST_NO[:,1,:,:]**2).sum(-2)).T,color='#FFC107',linewidth=0.1)
mean_dist_no, =ax[1].plot(np.median(np.sqrt((DIST_NO[:,1,:,:]**2).sum(-2)),0).T,color='#FFC107')

ax[1].legend([mean_dist_cross,mean_dist_skew,mean_dist_no],['Cross', 'Skew','Non-Unitary'])
ax[1].set(ylim=(0,.01))
ax[1].set_xlabel('Batch')
ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.suptitle('Experimental results')
ax[0].set_xticks([0,5,10])

"==========================================="

plt.show()

