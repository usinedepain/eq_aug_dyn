import os
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets
import torchvision.transforms as transforms
import networks
import mymse
from trainer_local import train
from tester_local import test
from tqdm import trange

# Code for running experiments locally

####################################################

# LOWER THESE PARAMETERS TO REDUCE COMPUTATIONAL TIME!
# NO_EXPERIMENTS NEEDS TO BE DIVISIBLE BY 3.
NO_EXPERIMENTS = 90 # Set the no. experiments. This is the size of the task array
EPOCHS = 250 # Set no. epochs for training equivariantly
AUG_EPOCHS = 10 # Set no. epochs for training with augmentation

####################################################

# Initialize task array
TASK_ID_INT = np.zeros(NO_EXPERIMENTS)
for i in range(NO_EXPERIMENTS):
    TASK_ID_INT[i] = i

# Task id as integer
TASK_ID_INT = TASK_ID_INT.astype(int)
# Get the task id string
TASK_ID = TASK_ID_INT.astype(str)
NO_CROSS = NO_EXPERIMENTS/3 # No. of experiments with cross-shaped support
BATCH_SIZE = 100 # Set batch size
TEST_BATCH_SIZE = 50 # Batch size for testing
PATH = os.getcwd() # Get current directory
PATH1 = os.path.join(PATH, r'EqvAugDyn/Local_Testing/') # create new directory name
PATH2 = os.path.join(PATH, r'data/') # create new directory name
if not os.path.isdir(PATH1): # if the directory does not already exist
    os.makedirs(PATH1) # make a new directory
else:
    pass
if not os.path.isdir(PATH2): # if the directory does not already exist
    os.mkdir(PATH2) # make a new directory
else:
    pass
LOAD_ROOT = './data/' # Root for loading dataset
SAVE_ROOT = './EqvAugDyn/Local_Testing/' # Root for saving data
LEARNING_RATE = 5e-4 # Set learning rate
WEIGHT_DECAY = 0

FIRST_LAYER_OUT = 32 # Channels out of first layer
SECOND_LAYER_OUT = 64 # Channels out of second layer
THIRD_LAYER_OUT = 64 # Channels out of third layer
LOWEST_IMAGE_DIMENSION = 7 # Pooling lowers dimension of image down from 28x28 to 7x7
NO_CLASSES = 10
ORTHOGONAL = True

# Load MNIST and calculate mean and standard deviation
train_data = datasets.MNIST(
    root=LOAD_ROOT,
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
muhat = train_data.data.float().mean()/255
sigmahat = train_data.data.float().std()/255

# Create transforms for the data, including normalization w.r.t. mean and standard deviation
train_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(muhat,sigmahat)])
test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(muhat,sigmahat)])

# Load data with transforms applied
train_data = datasets.MNIST(
    root=LOAD_ROOT,
    train=True,
    download=True,
    transform=train_transform
)

trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = datasets.MNIST(
    root=LOAD_ROOT,
    train=False,
    download=True,
    transform=test_transform
)

testloader = DataLoader(test_data,batch_size=TEST_BATCH_SIZE,shuffle=False)


# Create augmented data set by concatenating rotations of base dataset
# by pi/half, pi, three-pi/half radians along with the base dataset.
def rot_pi_half(image):
    return torch.rot90(image, 1, [1,2])

def rot_pi(image):
    return torch.rot90(image, 2, [1,2])

def rot_three_pi_halfs(image):
    return torch.rot90(image, 3, [1,2])

train_pi_half_transform = transforms.Compose([transforms.ToTensor(),rot_pi_half,transforms.Normalize(muhat,sigmahat)])
train_pi_transform = transforms.Compose([transforms.ToTensor(),rot_pi,transforms.Normalize(muhat,sigmahat)])
train_three_pi_halfs_transform = transforms.Compose([transforms.ToTensor(),rot_three_pi_halfs,transforms.Normalize(muhat,sigmahat)])

train_pi_half_data = datasets.MNIST(
    root=LOAD_ROOT,
    train=True,
    download=True,
    transform=train_pi_half_transform
)
train_pi_data = datasets.MNIST(
    root=LOAD_ROOT,
    train=True,
    download=True,
    transform=train_pi_transform
)
train_three_pi_halfs_transform = datasets.MNIST(
    root=LOAD_ROOT,
    train=True,
    download=True,
    transform=train_three_pi_halfs_transform
)
train_aug_data = ConcatDataset([train_data,train_pi_half_data,train_pi_data,train_three_pi_halfs_transform])

augtrainloader = DataLoader(train_aug_data, batch_size=BATCH_SIZE, shuffle=True)

# Create cross-shaped kernel
supp=torch.zeros(3,3)
supp[0,1]=1
supp[1,0]=1
supp[1,1]=1
supp[1,2]=1
supp[2,1]=1

# Create skewed kernel
supp2=torch.zeros(3,3)
supp2[0,1]=1
supp2[1,0]=1
supp2[0,0]=1
supp2[1,2]=1
supp2[2,1]=1

# Give device as 'cuda'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Give criterion as MSE, from custom script
criterion = mymse.mse

# Initialize arrays to store data
eqv_gradient_norm_array = np.zeros((EPOCHS,len(trainloader)))
eqv_testing_loss_array = np.zeros(EPOCHS)
eqv_testing_acc_array = np.zeros(EPOCHS)
eqv_training_loss_array = np.zeros((EPOCHS,len(trainloader)))
eqv_training_acc_array = np.zeros(EPOCHS)
aug_gradient_norm_array =  np.zeros((AUG_EPOCHS,len(augtrainloader)))
aug_testing_loss_array = np.zeros(AUG_EPOCHS)
aug_testing_acc_array = np.zeros(AUG_EPOCHS)
aug_training_loss_array = np.zeros((AUG_EPOCHS,len(augtrainloader)))
aug_training_acc_array = np.zeros(AUG_EPOCHS)
aug_training_distance_array = np.zeros((AUG_EPOCHS,4,len(augtrainloader)))
for i in range(NO_EXPERIMENTS):
    if TASK_ID_INT[i] < NO_CROSS:
        # If the task belongs to the first third of the array
        # initialize the equivariant CNN with cross-shaped kernel.
        # Also set the correct filename for saving results.

        model = networks.eqvConvNet(supp,LOWEST_IMAGE_DIMENSION,NO_CLASSES,hidden=(FIRST_LAYER_OUT,SECOND_LAYER_OUT,THIRD_LAYER_OUT)) # Initialize equivariant model
        model = model.to(device)
        FILE_NAME = 'cross'

        # Initilize the non-equivariant model
        augmodel = networks.omegaConvNet(supp,LOWEST_IMAGE_DIMENSION,NO_CLASSES,hidden=(FIRST_LAYER_OUT,SECOND_LAYER_OUT,THIRD_LAYER_OUT),orthogonal=ORTHOGONAL)
        augmodel = augmodel.to(device)

    elif NO_CROSS - 1 < TASK_ID_INT[i] < 2*NO_CROSS:
        # For the middle third of the task array, initialize the equivariant CNN with a skewed kernel.
        # Also set the filename.

        model = networks.eqvConvNet(supp2,LOWEST_IMAGE_DIMENSION,NO_CLASSES,hidden=(FIRST_LAYER_OUT,SECOND_LAYER_OUT,THIRD_LAYER_OUT)) # Initialize equivariant model
        model = model.to(device)   
        FILE_NAME = 'skew'

        # Initilize the non-equivariant model
        augmodel = networks.omegaConvNet(supp2,LOWEST_IMAGE_DIMENSION,NO_CLASSES,hidden=(FIRST_LAYER_OUT,SECOND_LAYER_OUT,THIRD_LAYER_OUT),orthogonal=ORTHOGONAL)
        augmodel = augmodel.to(device)

    else:
        # If the task belongs to the last third of the array
        # initialize the equivariant CNN with cross-shaped kernel,
        # but use a non-orthogonal basis for the kernel when augmenting.
        # Also set filename.
        ORTHOGONAL = False

        model = networks.eqvConvNet(supp,LOWEST_IMAGE_DIMENSION,NO_CLASSES,hidden=(FIRST_LAYER_OUT,SECOND_LAYER_OUT,THIRD_LAYER_OUT)) # Initialize equivariant model
        model = model.to(device)
        FILE_NAME = 'cross_non_orthogonal'

        # Initilize the non-equivariant model.
        augmodel = networks.omegaConvNet(supp,LOWEST_IMAGE_DIMENSION,NO_CLASSES,hidden=(FIRST_LAYER_OUT,SECOND_LAYER_OUT,THIRD_LAYER_OUT),orthogonal=ORTHOGONAL)
        augmodel = augmodel.to(device)

    # Give optimizer as SGD with weight decay on or off (=WEIGHT_DECAY)
    optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY) 
    for epoch in trange(EPOCHS, desc='Epoch progress', leave=False):
        # Train and test for a number of epochs

        training_loss, training_acc, gradient_norm, distance = train(model,trainloader,optimizer,criterion,device)
        testing_loss, testing_acc = test(model,testloader,criterion,device)

        # Store data
        eqv_gradient_norm_array[epoch,:] = gradient_norm
        eqv_testing_loss_array[epoch] = testing_loss
        eqv_testing_acc_array[epoch] = testing_acc
        eqv_training_loss_array[epoch,:] = training_loss
        eqv_training_acc_array[epoch] = training_acc
        

    # Save stored data
    np.save(os.path.join(SAVE_ROOT,'eqv_gradient_norm_'+FILE_NAME+TASK_ID[i]),eqv_gradient_norm_array)
    np.save(os.path.join(SAVE_ROOT,'eqv_testing_loss_'+FILE_NAME+TASK_ID[i]),eqv_testing_loss_array)
    np.save(os.path.join(SAVE_ROOT,'eqv_testing_acc_'+FILE_NAME+TASK_ID[i]),eqv_testing_acc_array)
    np.save(os.path.join(SAVE_ROOT,'eqv_training_loss_'+FILE_NAME+TASK_ID[i]),eqv_training_loss_array)
    np.save(os.path.join(SAVE_ROOT,'eqv_training_acc_'+FILE_NAME+TASK_ID[i]),eqv_training_acc_array)

    augmodel.load_weights(model) # Load the weights from the equivariant model

    # Give optimizer as SGD with weight decay on or off (=WEIGHT_DECAY), and halved learning rate
    optimizer = torch.optim.SGD(augmodel.parameters(),lr=LEARNING_RATE/2,weight_decay=WEIGHT_DECAY) 
    for epoch in trange(AUG_EPOCHS, desc='Epoch progress', leave=False):
        # Train and test for a number of epochs on augmented data

        training_loss, training_acc, gradient_norm, distance = train(augmodel,augtrainloader,optimizer,criterion,device)
        testing_loss, testing_acc = test(augmodel,testloader,criterion,device)

        # Store data
        aug_gradient_norm_array[epoch,:] = gradient_norm
        aug_testing_loss_array[epoch] = testing_loss
        aug_testing_acc_array[epoch] = testing_acc
        aug_training_loss_array[epoch,:] = training_loss
        aug_training_acc_array[epoch] = training_acc
        aug_training_distance_array[epoch,:,:] = distance
        
    # Save stored data
    np.save(os.path.join(SAVE_ROOT,'aug_gradient_norm_'+FILE_NAME+TASK_ID[i]),aug_gradient_norm_array)
    np.save(os.path.join(SAVE_ROOT,'aug_testing_loss_'+FILE_NAME+TASK_ID[i]),aug_testing_loss_array)
    np.save(os.path.join(SAVE_ROOT,'aug_testing_acc_'+FILE_NAME+TASK_ID[i]),aug_testing_acc_array)
    np.save(os.path.join(SAVE_ROOT,'aug_training_loss_'+FILE_NAME+TASK_ID[i]),aug_training_loss_array)
    np.save(os.path.join(SAVE_ROOT,'aug_training_acc_'+FILE_NAME+TASK_ID[i]),aug_training_acc_array)
    np.save(os.path.join(SAVE_ROOT,'aug_distance_'+FILE_NAME+TASK_ID[i]),aug_training_distance_array)

    print('Done with '+FILE_NAME)