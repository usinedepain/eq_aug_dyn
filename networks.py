import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class eqvConv(nn.Module):
    # Custom convolutional layer with a 3 x 3 kernel with given support, omega.
    # Layer is equivariant to the action of C_4.

    def __init__(self, in_channels: int, out_channels: int, omega):
        super().__init__()

        self.pattern = (omega==0).nonzero() # store indexes outside the support omega
        self.out_channels = out_channels # In channels
        self.in_channels = in_channels # Out channels

        # Create masks corresponding to the middle, sides, and corners of the kernel
        mid_mask = torch.zeros(3,3)
        mid_mask[1,1] = 1
        side_mask = torch.zeros(3,3)
        side_mask[0,1] = 1/2
        side_mask[1,0] = 1/2
        side_mask[1,2] = 1/2
        side_mask[2,1] = 1/2
        corner_mask = torch.zeros(3,3)
        corner_mask[0,0] = 1/2
        corner_mask[0,2] = 1/2
        corner_mask[2,0] = 1/2
        corner_mask[2,2] = 1/2

        # Register masks
        self.register_buffer('mid_mask',mid_mask)
        self.register_buffer('side_mask',side_mask)
        self.register_buffer('corner_mask',corner_mask)

        # Initialize the parameters for the middle, side, and corner values.
        self.nullsetzen()
        j=0
        for b in self.named_buffers():
            j+=1
        i=0
        for n, b in self.named_buffers():
            if n != 'mid':
                i+=1
        if i==j:
            self.mid = nn.Parameter(torch.empty(out_channels,in_channels))
            torch.nn.init.kaiming_uniform_(self.mid, a=math.sqrt(5))
        i=0
        for n, b in self.named_buffers():
            if n != 'side':
                i+=1
        if i==j:
            self.side = nn.Parameter(torch.empty(out_channels,in_channels))
            torch.nn.init.kaiming_uniform_(self.side, a=math.sqrt(5))
            self.side = nn.Parameter(self.side * 2)
        i=0
        for n, b in self.named_buffers():
            if n != 'corner':
                i+=1
        if i==j:
            self.corner = nn.Parameter(torch.empty(out_channels,in_channels))
            torch.nn.init.kaiming_uniform_(self.corner, a=math.sqrt(5))
            self.corner = nn.Parameter(self.corner * 2)
        
    def nullsetzen(self):
        # Zeroes out either corners, sides, or middle values depending on the convolutional kernel's
        # support, omega, in order to guarantee equivariance. 
        out_channels = self.out_channels
        in_channels = self.in_channels
        for i in range(self.pattern.shape[0]):
            if (torch.equal(self.pattern[i,:],torch.Tensor([0,0]))) | (torch.equal(self.pattern[i,:],torch.Tensor([0,2]))) | (torch.equal(self.pattern[i,:],torch.Tensor([2,0]))) | (torch.equal(self.pattern[i,:],torch.Tensor([2,2]))):
                corner = torch.zeros(out_channels,in_channels) # Zero corners
                self.register_buffer('corner', corner) # Hide corners from autograd

            elif (torch.equal(self.pattern[i,:],torch.Tensor([0,1]))) | (torch.equal(self.pattern[i,:],torch.Tensor([1,0]))) | (torch.equal(self.pattern[i,:],torch.Tensor([1,2]))) | (torch.equal(self.pattern[i,:],torch.Tensor([2,1]))):
                side = torch.zeros(out_channels,in_channels) # Zero sides
                self.register_buffer('side',side) # Hide sides from autograd

            elif torch.equal(self.pattern[i,:],torch.Tensor([1,1])):
                mid = torch.zeros(out_channels,in_channels) # Zero middle
                self.register_buffer('mid',mid) # Hide middle from autograd

            else:
                pass

    def calculate_weight_tensor(self):
        # Calculates equivariant weights with appropriate dimensions for nn.functional.conv2d.

        # Stacks parameters and masks
        stacked_param = torch.stack((self.mid,self.side,self.corner),dim=2)
        stacked_mask = torch.stack((self.mid_mask,self.side_mask,self.corner_mask), dim=0)
        
        # Unsqueezes to match dimensions to make broadcasting possible
        stacked_param = torch.unsqueeze(torch.unsqueeze(stacked_param,-1),-1)
        stacked_mask = torch.unsqueeze(torch.unsqueeze(stacked_mask,0),0)

        # Adds the products of the elementwise multiplications of mid (m), side (s), and corner (c)
        # with their respective masks together to form an equivariant weight tensor for convolution
        # where the last two dimensions look like
        # c  s  c        0  s  0         c  0  c         0  0  0
        # s  m  s   =    s  0  s    +    0  0  0    +    0  m  0
        # c  s  c        0  s  0         c  0  c         0  0  0
        weight = torch.sum(torch.mul(stacked_param, stacked_mask),dim=2)

        return weight

    def forward(self,x):
        weight = self.calculate_weight_tensor()
        return F.conv2d(x, weight,bias=None,stride=1,padding=1)
    
class eqvlinear(nn.Module):
    # Custom linear layer with weights equivariant to the action of C_4.

    def __init__(self, channels: int, im_dim: int, out_dim: int):
        super().__init__()

        # Stores some dimensions which are used for creating and reshaping the weights
        self.channels = channels # the channels out of the previous convolutional layer
        self.imdim = im_dim # row dimension of the square image coming from the last pooling (=7 for MNIST)
        self.outdim = out_dim # out dimension of the linear layer (=10 for MNIST)
        
        # Create a stack of masks corresponding to the values of the "corner matrices" of the weight tensor
        idx_matrix = torch.arange(np.ceil(self.imdim/2).astype('int')*np.floor(self.imdim/2).astype('int'))
        idx_matrix = torch.reshape(idx_matrix,(self.imdim-np.ceil(self.imdim/2).astype('int'),self.imdim-np.floor(self.imdim/2).astype('int')))
        corner_mask = torch.zeros(np.ceil(self.imdim/2).astype('int')*np.floor(self.imdim/2).astype('int'),self.imdim,self.imdim)
        for i in range(self.imdim-np.ceil(self.imdim/2).astype('int')):
            for j in range(self.imdim-np.floor(self.imdim/2).astype('int')):
                k = idx_matrix[i,j]
                corner_mask[k,i,j]=1/2

        self.register_buffer('corner_mask',corner_mask) # Register mask

        # If image dimension is odd we add a mask for middle values of weight tensor
        if np.ceil(self.imdim/2).astype('int')!=np.floor(self.imdim/2).astype('int'):
            mid_mask = torch.zeros(self.imdim,self.imdim)
            mid_mask[np.floor(self.imdim/2).astype('int'),np.floor(self.imdim/2).astype('int')]=1
            self.register_buffer('mid_mask',mid_mask) # Register mask

        # Initialize parameters for "corner matrices" of the weight tensor.
        # Ceil and floor are used in case image dimension is odd.
        self.corner_matrix = nn.Parameter(torch.empty(self.outdim,self.channels,np.ceil(self.imdim/2).astype('int'),np.floor(self.imdim/2).astype('int')))
        torch.nn.init.kaiming_uniform_(self.corner_matrix, a=math.sqrt(5))
        self.corner_matrix = nn.Parameter(self.corner_matrix * 2)

        # If image dimension is odd we add a middle value as a parameter
        if np.ceil(self.imdim/2).astype('int')!=np.floor(self.imdim/2).astype('int'):
            self.mid = nn.Parameter(torch.empty(self.outdim,self.channels))
            torch.nn.init.kaiming_uniform_(self.mid, a=math.sqrt(5))

    def calculate_weight_tensor(self):
        # Calculates equivariant weights with appropriate dimensions for nn.functional.linear.

        # Corner matrices (flattened)
        corner_param = torch.reshape(self.corner_matrix,(self.outdim,self.channels,np.ceil(self.imdim/2).astype('int')*np.floor(self.imdim/2).astype('int')))
        corner_mask = self.corner_mask
        
        # Unsqueezes to match dimensions to make broadcasting possible
        corner_param = torch.unsqueeze(torch.unsqueeze(corner_param,-1),-1)
        corner_mask = torch.unsqueeze(torch.unsqueeze(corner_mask,0),0)


        # Calculate corner weights
        weight = torch.sum(torch.mul(corner_param,corner_mask),dim=2)
        weight = weight + torch.rot90(weight,1,[2,3]) + torch.rot90(weight,2,[2,3]) + torch.rot90(weight,3,[2,3])

        # Middle values if image dimension is odd
        if np.ceil(self.imdim/2).astype('int')!=np.floor(self.imdim/2).astype('int'):
            mid_param = self.mid
            mid_mask = self.mid_mask

            # Unsqueezes to match dimensions to make broadcasting possible
            mid_param = torch.unsqueeze(torch.unsqueeze(mid_param,-1),-1)
            mid_mask = torch.unsqueeze(torch.unsqueeze(mid_mask,0),0)

            # Calculates and adds middle weights
            weight = weight + torch.mul(mid_param,mid_mask)

        # Reshape weight into form suitable for nn.functional.linear
        weight = torch.reshape(weight,(self.outdim,self.channels * self.imdim * self.imdim))

        return weight

    def forward(self,x):
        weight = self.calculate_weight_tensor()
        return F.linear(x,weight,bias=None)

class eqvConvNet(nn.Module):
    # Convolutional neural network, equivariant to the action of C_4.
    def __init__(self,omega,lowest_im_dim,out_dim,hidden=(32,64,64)):
        super().__init__()
        self.conv1 = eqvConv(1,hidden[0],omega)
        self.conv2 = eqvConv(hidden[0],hidden[1],omega)
        self.conv3 = eqvConv(hidden[1],hidden[2],omega)
        self.lin = eqvlinear(hidden[2],lowest_im_dim,out_dim)
        
        self.pool = nn.AvgPool2d(2)
        self.activate = nn.Tanh()
        self.bn1 = nn.LayerNorm(normalized_shape=(14,14),elementwise_affine=False)
        self.bn2 = nn.LayerNorm(normalized_shape=(7,7),elementwise_affine=False)

    def is_equivariant(self):
        return True
    def forward(self,x):
        x = self.activate(self.pool(self.conv1(x)))
        x = self.bn1(x)
        x = self.activate(self.pool(self.conv2(x)))
        x = self.bn2(x)
        x = self.activate(self.conv3(x))
        x = self.bn2(x)
        h = x.view(x.shape[0], -1)
        x = self.lin(h)
        return x, h
    
class omegaConv(nn.Module):
    # Non-equivariant version of eqvConv
    def __init__(self, in_channels: int, out_channels: int, omega, orthogonal=True):
        super().__init__()

        # Detect the pattern of the support
        pattern = omega.nonzero()
        self.pattern = pattern
        
        self.orthogonal=orthogonal

        # Create and register buffer for stacked support mask
        stacked_mask = torch.zeros(pattern.shape[0],3,3)

        for i in range(pattern.shape[0]):
            stacked_mask[i,pattern[i,:][0],pattern[i,:][1]] = 1
        # If orthogonal is set to false, generate a non-orthogonal basis instead.
        if not orthogonal:
            basis = torch.zeros(pattern.shape[0],3,3)
            coeff = torch.randn((pattern.shape[0],pattern.shape[0]))
            for i in range(pattern.shape[0]):
                for j in range(pattern.shape[0]):
                    basis[i,:,:] += coeff[i,j]*stacked_mask[j,:,:]
                basis[i,:,:] = basis[i,:,:] / torch.norm(basis[i,:,:])
            self.register_buffer('stacked_mask',basis)
        else:
            self.register_buffer('stacked_mask',stacked_mask)

        # Initialize weight tensor
        self.weight = nn.Parameter(torch.empty(out_channels,in_channels,pattern.shape[0]))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def load_weights(self,layer):
        # Function to load model weights
        weight = torch.zeros_like(self.weight)
        with torch.no_grad():
            pattern = self.pattern
            coeffs = layer.calculate_weight_tensor()
            if self.orthogonal:
                for i in range(pattern.shape[0]):
                    weight[:,:,i] = coeffs[:,:,pattern[i,:][0],pattern[i,:][1]]
            else:
                L = self.stacked_mask.flatten(-2,-1).transpose(-2,-1)
                phi = coeffs.flatten(-2,-1)
                w = torch.einsum('ij,bli->blj',L,phi)
                weight = torch.linalg.solve(L.T@L,w.transpose(-2,-1))
                weight = torch.transpose(weight,-2,-1)
            self.weight = nn.Parameter(weight)

    def calculate_weight_tensor(self):
        # Calculates weight tensor with given support

        # Parameters and masks
        weight = self.weight
        mask = self.stacked_mask

        # Unsqueeze to match dimensions
        weight = torch.unsqueeze(torch.unsqueeze(weight,-1),-1)
        mask = torch.unsqueeze(torch.unsqueeze(mask,0),0)

        # Calculate weight
        weight = torch.sum(torch.mul(weight,mask),dim=2)

        return weight
    
    def projection(self):
        # Calculates projection onto the equivariant subspace
        weight = self.calculate_weight_tensor()
        proj = weight + torch.rot90(weight,1,[2,3]) + torch.rot90(weight,2,[2,3]) + torch.rot90(weight,3,[2,3])
        proj /= 4
        mask = self.stacked_mask
        mask = torch.unsqueeze(torch.unsqueeze(mask,0),0)
        mask = torch.sum(torch.abs(mask),dim=2)
        mask[mask!=0] = 1
        proj = proj*mask
        proj = proj*torch.rot90(mask,1,(-2,-1))
        proj = proj*torch.rot90(mask,2,(-2,-1))
        proj = proj*torch.rot90(mask,3,(-2,-1))
        return proj

    def forward(self,x):
        weight = self.calculate_weight_tensor()
        return F.conv2d(x,weight,bias=None,stride=1,padding=1)
        
class omegaConvNet(nn.Module):
    # Convolutional neural network, with a given support for the kernel
    def __init__(self,omega,lowest_im_dim,out_dim,hidden=(32,64,64),orthogonal=True):
        super().__init__()
        self.lowest_im_dim = lowest_im_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.conv1 = omegaConv(1,hidden[0],omega,orthogonal)
        self.conv2 = omegaConv(hidden[0],hidden[1],omega,orthogonal)
        self.conv3 = omegaConv(hidden[1],hidden[2],omega,orthogonal)
        self.lin = nn.Linear(hidden[2]*lowest_im_dim*lowest_im_dim,out_dim,bias=False)
        self.pool = nn.AvgPool2d(2)
        self.activate = nn.Tanh()
        self.bn1 = nn.LayerNorm(normalized_shape=(14,14),elementwise_affine=False)
        self.bn2 = nn.LayerNorm(normalized_shape=(7,7),elementwise_affine=False)

    def is_equivariant(self):
        return False
    
    def load_weights(self, model):
        # Calls the weight loading function in each convolution layer and
        # loads the weights from the linear layer.
        with torch.no_grad():
            weight = model.lin.calculate_weight_tensor()
            self.lin.weight = nn.Parameter(weight)
            self.conv1.load_weights(model.conv1)
            self.conv2.load_weights(model.conv2)
            self.conv3.load_weights(model.conv3)

    def projection(self):
        # Calls the projection function in each convolution layer and
        # calculates the projection in the linear layer.
        proj_conv1 = self.conv1.projection()
        proj_conv2 = self.conv2.projection()
        proj_conv3 = self.conv3.projection()
        weight = self.lin.weight
        proj = torch.reshape(weight, (self.out_dim,self.hidden[2],self.lowest_im_dim,self.lowest_im_dim))
        proj = proj+torch.rot90(proj,1,[2,3])+torch.rot90(proj,2,[2,3])+torch.rot90(proj,3,[2,3])
        proj /= 4
        proj_lin = torch.reshape(proj, (self.out_dim, self.hidden[2]*self.lowest_im_dim*self.lowest_im_dim))
        return proj_conv1, proj_conv2, proj_conv3, proj_lin
    
    def calculate_weight_tensor(self):
        # Calls the weight calculation function in each convolution layer
        # and calculates the weights in the linear layer.
        weight_conv1 = self.conv1.calculate_weight_tensor()
        weight_conv2 = self.conv2.calculate_weight_tensor()
        weight_conv3 = self.conv3.calculate_weight_tensor()
        weight_lin = self.lin.weight
        return weight_conv1, weight_conv2, weight_conv3, weight_lin

    def forward(self,x):
        x = self.activate(self.pool(self.conv1(x)))
        x = self.bn1(x)
        x = self.activate(self.pool(self.conv2(x)))
        x = self.bn2(x)
        x = self.activate(self.conv3(x))
        x = self.bn2(x)
        h = x.view(x.shape[0], -1)
        x = self.lin(h)
        return x, h