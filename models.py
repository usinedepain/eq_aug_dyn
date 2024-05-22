import torch 
import numpy as np
from abc import abstractmethod

from torch.fft import fft2, ifft2, fft, ifft

from torch.fft import fft2, ifft2, fft, ifft # for convolution-related projections


class ProjectNet(torch.nn.Module):
    
    """
        class ProjectNet
        abstract class for neural networks that can be made equivariant through 
        projection of the linear layers.
        
        The structure of the representations on the intermediate layers are of the form
        
        (rho, rho, ... , rho, triv, ... , triv)
        
        or 
        
        (rho, rho, ..., rho)
        
        Therefore, we only need to be able to project 'standard' (rho -> rho)
        and 'invarizing' (rho -> triv) layers.
        
        ns gives the dimensions of the spaces on which rho is acting.
        
        If the model is of the former type, the implementing class needs to
            * call constructor of this class with with fully_connected=True
            * define own variables for the layers between spaces with trivial representation
        
        Methods:
            __init__(ns, device,scalar_output): basic constructor for initializing empty layers
                        classes inhereting from this class will need to specify 
                        * non-linearities: nonlin and last_lonlin
                        * n : 'parameters of' size of the representations
            forward(x): sends an x through the network
            copy: returns a new ProjectNet with the same data in the linear layers as self, but with 
                    computation graph independent of self.
            project_with_matrix: if implementation uses a projector matrix for project(A)-method,
                        this wrapper can be used to carry out the projection
            project_layers: projects all layers
            project_grads: projects gradients stored in the layers.
                        
        Abstract methods:
            reset: Function for initializing the linear layers
            project: function for projecting standard layers
            project_light: function for projecting invariazing layers
            
            
    """
    
    def __init__(self,ns, device = 'cpu', fully_connected = False):
        
        super(ProjectNet,self).__init__()
        self.layers = torch.nn.ParameterList()
        for k in range(len(ns)-1):
            layer = torch.nn.Parameter(torch.empty((ns[k+1],ns[k],1,1),dtype=torch.float,device=device))
            self.layers.append(layer)
            # the layer has size ns[k+1],ns[k],n,n
        
        if fully_connected:
            # if the architecture is fully connected, add invarizing layer.
            layer = torch.nn.Parameter(torch.empty((1,ns[-1],1,),dtype=torch.float,device=device)) 
            self.layers.append(layer)
        
        # correct sizes are decided in reset-method, in particular in dependence of the abstract dimension factor
        # reset method is called from constructor of class implementing the abstract class
        
        self.ns = ns
        
        self.device = device
        
        self.nonlin = torch.nn.Softplus(beta=1)
        self.last_nonlin = torch.nn.Softmax(dim=1)
        
        # Are trivial representations are present in the end of the intermediate spaces?
        self.fully_connected = fully_connected
        
    
        
    
    def forward(self,x):
            # wrapper for applying the forward of early layers
            # 'fully connected heads' are applied in forward models of
            # class implementing the abstract class.
            
            if x.dim()>3:
                x = x.flatten(-2,-1)

            for k in range(len(self.layers)-1):
                layer = self.layers[k]
                x = torch.einsum('ijkl,bjl->bik',layer,x)
                x = self.nonlin(x)
          
            x = torch.einsum('ijkl,bjl->bik',self.layers[-1],x)
            x = self.last_nonlin(x)
            return x
   
    @abstractmethod
    def reset(self):
        
        # reset all the layers
        # abstract method.
        
        pass
     
         
              
    def copy(self):
         
        # copy the network, including the values of the early weights.
        
         with torch.no_grad():
             cpy = self.__class__(self.ns,self.n, self.device,fully_connected= self.fully_connected)
         
             for (k,layer) in enumerate(self.layers):
                 cpy.layers[k].data = self.layers[k].data.clone().detach().requires_grad_(True)
          
         return cpy 
     
   
    @abstractmethod
    def project(self,A):
        
        # function for projecting 'standard' layers
        # abstract method
        
        pass

       
    
    def project_with_matrix(self,A):
        
        # if the net operates with a projection matrix projmat, this routine can be used to implement project
        
        coeffs = A.flatten(-2,-1)@self.projmat.T
        return coeffs @ self.projmat 
        

    
    @abstractmethod
    def project_light(self,A):
        # for projecting invarizing layers
        pass
       
        
    def project_layers(self):
        # wrapper for projecting all layers
        # note that also when self.fully_connected = True , the layers between the late spaces
        # do not need to be projected
        
        for k in range(len(self.layers)-1):
            self.layers[k].data = self.project(self.layers[k].data)
        if self.fully_connected:
            self.layers[-1].data = self.project_light(self.layers[-1].data)
        else:
            self.layers[-1].data = self.project(self.layers[-1].data)
            
    def project_grads(self):
        # wrapper for projecting all gradients
        # note that also when self.fully_connected = True , the layers between the late spaces
        # do not need to be projected
        
        for k in range(len(self.layers)-1):
            self.layers[k].grad = self.project(self.layers[k].grad)
                 
        if self.fully_connected:
            self.layers[-1].data = self.project_light(self.layers[-1].data)
        else:
            self.layers[-1].data = self.project(self.layers[-1].data)

class PermutationAdjNet(ProjectNet):

    """
        class PermutationAdjNet
            A Net for processing R^{n,n}- features in a permutation-equivariant
            manner.
            
            if fully_connected =True, ms needs to be specified as the dimensions of the intermediate spaces
            
    """
    def __init__(self,ns, n, device = 'cpu', ms =[], fully_connected=False):
        super(PermutationAdjNet,self).__init__(ns,device,fully_connected)
        
        self.n = n
        
        # non-linearities
        self.nonlin = torch.nn.LeakyReLU()
        self.last_nonlin = torch.nn.LeakyReLU()
        self.fc_nonlin = torch.nn.Sigmoid()
        self.fully_connected=fully_connected
        
        
        if fully_connected:
            self.ms = ms
            # the final output space is one-dimensional
            ms.append(1)
            
            #fully connected layers
            self.fclayers = torch.nn.ParameterList()
            self.bn = torch.nn.LayerNorm(ms[0], device= self.device,elementwise_affine=False) #
        
        # initialize the matrices with normal random coefficients
        self.reset()
        
        # pre-calculate projection matrix
        self.projmat = self.projector(self.n)
        self.projmat = self.projmat.to(device)
         
        
    def copy(self):
        
        # copy the early layers
        cpy = super().copy()
        
        # handle layers connected to full connection if appropriate
        cpy.fully_connected = self.fully_connected
        if self.fully_connected:
            cpy.ms = self.ms
            cpy.fclayers = torch.nn.ParameterList()
            cpy.fc_nonlin = self.fc_nonlin
            cpy.bn = torch.nn.LayerNorm(self.ms[0], device = self.device,elementwise_affine=False)
            
            for k in range(1,len(self.fclayers)+1):
                lin = torch.nn.Parameter(torch.zeros_like(self.fclayers[k-1].data))
                cpy.fclayers.append(lin)
                cpy.fclayers[k-1].data = self.fclayers[k-1].data.clone().detach().requires_grad_(True)
                
        cpy.last_nonlin = self.last_nonlin
        

         
        return cpy
        
    def forward(self,x):

        x = x-.5 # normalize
        x = x.reshape(x.shape[0],x.shape[1],self.n**2)
        
        # early layers are applied 
        y = super().forward(x)
        
        # apply late layers
        if self.fully_connected:
            #batch_normalization
            y = y.flatten(-2,-1)
            y = self.bn(y)
            for k, fclayer in enumerate(self.fclayers):
                y = torch.einsum('ji,bj->bi',fclayer,y)

                
                if k < len(self.fclayers)-1:
                    y = self.nonlin(y)
                else:
                    # final non-linearity is different
                    y = self.fc_nonlin(y)
            return y.flatten(-2,-1)
        else:
            return y
        
        
    def reset(self):
        # reset the weights according to a Gaussian distribution.
        for k in range(len(self.ns)-1):
            self.layers[k].data = 1/(self.n*np.sqrt(self.ns[k]*self.ns[k+1]))*torch.randn(self.ns[k+1],self.ns[k],self.n**2,self.n**2)
            
            
        if self.fully_connected:
            self.layers[-1].data = 1/(np.sqrt(self.ms[0]))*(torch.randn(self.ms[0],self.ns[-1],1,1)*torch.ones(self.ms[0],self.ns[-1],1,self.n**2)).to(self.device)
            for k in range(1, len(self.ms)):
                lin = torch.nn.Parameter(torch.randn(self.ms[k-1],self.ms[k],device = self.device))
                lin.data = lin.data*self.n/np.sqrt(self.ms[k-1]*self.ms[k-1])
                self.fclayers.append(lin)
            

        
    @staticmethod
    def projector(n):
        
        """
           Explicitly define the projection matrix, using the results of 
               Haggai Maron, Heli Ben-Hamu, Nadav Shamir, and Yaron Lipman. "Invariant and equivariant graph networks", ICLR 2019.
        """
        
        basis = np.zeros((15,n,n,n,n))
        
        for i in range(n):
            basis[0,i,i,i,i] = 1
            for j in range(n):
                if not i==j:
                    basis[1,i,i,i,j] = 1
                    basis[2,i,i,j,i] = 1
                    basis[3,i,j,i,i] = 1
                    basis[4,j,i,i,i] = 1
                    
                    basis[5,i,i,j,j] = 1
                    basis[6,i,j,i,j] = 1
                    basis[7,j,i,i,j] = 1

                    for k in range(n):
                        if k!=i and k!=j:
                            basis[8,i,i,j,k] = 1
                            basis[9,i,j,i,k] = 1
                            basis[10,i,j,k,i] = 1
                            basis[11,j,i,i,k] = 1
                            basis[12,j,i,k,i] = 1
                            basis[13,j,k,i,i] = 1
                            
                            for l in range(n):
                                if l!=i and l!=j and l!=k:
                                    basis[14,i,j,k,l] = 1
            
            
        basis = basis.reshape(15,n**4)
            
        nrms = np.sqrt(basis.sum(1))
            
        basis = basis/nrms[:,None]
        return torch.tensor(basis,dtype=torch.float)
        
        
    ## functions for projecting
    def project(self,A):
        
       B = super().project_with_matrix(A)
       
       return B.reshape(B.shape[0],B.shape[1],self.n**2,self.n**2)
    
    def project_light(self,A):
        
        """ 
            Projections of layers from R^n \otimes R^n to R. See again 
                Haggai Maron, Heli Ben-Hamu, Nadav Shamir, and Yaron Lipman. "Invariant and equivariant graph networks", ICLR 2019.
            for explaination of the inner workings of this projection function.
        """
        
        n = self.n
        A = A.reshape(A.shape[0],self.ns[-1],n,n)
        trace = 1/(1-1/n)*A.diagonal(dim1=-2,dim2=-1).mean(-1)
        mean  = 1/(1-1/n)*A.mean((-2,-1),keepdim=True)
        
        id = torch.eye(n, device = A.device)
        
        A= mean -1/n*trace[:,:,None,None]+ (trace[:,:,None,None]-mean)*id[None,None,:,:]
        
        return A.reshape(A.shape[0],self.ns[-1],1,n**2)
    



    
class  TranslationNet(ProjectNet):

    """
        class TranslationNet
            A Net for processing R^{n,n}- features in a translation-equivariant
            manner.
    """   
    
    def __init__(self,ns,n, device = 'cpu', fully_connected=False, ms = []):
        super(TranslationNet,self).__init__(ns, device, fully_connected)
        
        self.n = n
        
        # non-linearities
        self.nonlin = torch.nn.LeakyReLU()
        if fully_connected:
            self.fc_nonlin = torch.nn.Softmax(dim=1)
            self.last_nonlin = torch.nn.Tanh()
        else:
            self.last_nonlin = torch.nn.Softmax(dim=1)
        
        # extra fully connected layers
        self.fully_connected = fully_connected
        
        if fully_connected:
            self.ms = ms
            ms.append(10)
            self.fclayers = torch.nn.ParameterList()

            self.bn = torch.nn.LayerNorm(ms[0], device= self.device, elementwise_affine=False) #
        
        # initialize the matrices with normal random coefficients
        self.reset()
        
        # pre-calculate projection matrix
        self.projmat = self.projector(self.n)
        self.projmat = self.projmat.to(device)
        q, _ = torch.linalg.qr(self.projmat.T)    # due to numerical issues, self.projmat is not orthogonal after
        self.projmat = q.T                        # transferring it to cuda. this mitigates this
                                                   
        
    def forward(self,x):

        #x = (x-x.mean((-2,-1),keepdim=True)) # normalize
        x = x-.5
        
        # the early layers are handled by the super class
        y = super().forward(x)
        
        # if needed, carry out the final layers
        if self.fully_connected:
            y = y.flatten(-2,-1)
            y = self.bn(y)
            for k, fclayer in enumerate(self.fclayers):
                y = torch.einsum('ji,bj->bi',fclayer,y)

                # non-linearity of final layer is different.
                if k < len(self.fclayers)-1:
                    y = self.nonlin(y)
                else:
                    y = self.fc_nonlin(y)
            return y
        else:
            return y

 

    def copy(self):
        #copy all variables, and values of the mlp
        cpy = super().copy()
        cpy.fclayers = torch.nn.ParameterList()
        cpy.fully_connected = self.fully_connected
        if self.fully_connected:
            cpy.ms = self.ms
            cpy.fc_nonlin = self.fc_nonlin
            cpy.bn = torch.nn.LayerNorm(cpy.ms[0], device= self.device,elementwise_affine=False)
        cpy.last_nonlin = self.last_nonlin
        
        
        if self.fully_connected:
            for k in range(1,len(self.fclayers)+1):
                lin = torch.nn.Parameter(torch.zeros_like(self.fclayers[k-1].data))
                cpy.fclayers.append(lin)
                cpy.fclayers[k-1].data = self.fclayers[k-1].data.clone().detach().requires_grad_(True)
        
        return cpy
    
    def reset(self):
        
        # reset all layers with Gaussian weights.
        for k in range(len(self.layers)-1):
            self.layers[k].data = 1/(self.n*np.sqrt(self.ns[k]))*torch.randn(self.ns[k+1],self.ns[k],self.n**2,self.n**2).to(self.device)
           
            

        if self.fully_connected:
            self.layers[-1].data = 1/(np.sqrt(self.ms[0]))*(torch.randn(self.ms[0],self.ns[-1],1,1)*torch.ones(self.ms[0],self.ns[-1],1,self.n**2)).to(self.device)
            for k in range(1, len(self.ms)):
                lin = torch.nn.Parameter(torch.randn(self.ms[k-1],self.ms[k],device = self.device))
                lin.data = lin.data*self.n/np.sqrt(self.ms[k-1]*self.ms[k-1])
                self.fclayers.append(lin)

    
    @staticmethod
    def projector(n):
        
        # define the projection matrix corresponding to the basis $C^{k\ell}$ defined in the appendix of the paper.
        basis = np.zeros((n,n,n,n,n,n))
        
        
        for i in range(n):           
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        basis[i,j,k,l,(k+i-1)%n,(l+j-1)%n]=1.0
            
            
        basis = basis.reshape(n**2,n**4)
            
        nrms = np.sqrt(basis.sum(1))
            
        basis = basis/nrms[:,None]
        return torch.tensor(basis,dtype=torch.float)    
    
    def project(self,A):
        
       m1 = A.shape[0]
       m2 = A.shape[1]

       # calculate F^TAF, where F is Fourier operator
       A = ifft2(fft2(A.reshape(m1,m2,self.n,self.n,self.n,self.n),dim=(-2,-1)),dim=(-4,-3))
       
       # extract diagonal DA
       A = torch.diag_embed(torch.diagonal(A.reshape(m1,m2,self.n**2,self.n**2),dim1=-1,dim2=-2))
       
       # Calculate F*DA*F^T and return
       A = fft2(ifft2((A).reshape(m1,m2,self.n,self.n,self.n,self.n),dim=(-2,-1)),dim=(-4,-3))
       
       return torch.real(A.reshape(m1,m2,self.n**2,self.n**2)).contiguous()
    
    def project_light(self,A):
        # To project the invarizing layers, we only need to take the mean.
        return A.mean(-1,keepdim=True)*torch.ones_like(A)  

class OneDTranslationNet(TranslationNet):
    
    """
        class TranslationNet
            A Net for processing R^{n,n}- features in a manner 
            equivariant to translations in x-direction.
    """  
    
    def __init__(self,ns,n, device = 'cpu', fully_connected=False, ms = []):
        super(OneDTranslationNet,self).__init__(ns, n, device, fully_connected)
        
    
    def project(self,A):

        
        m1 = A.shape[0]
        m2 = A.shape[1]
        
        # calculate F^TAF, where F is Fourier operator in one direction
        A = ifft(fft(A.reshape(m1,m2,self.n,self.n,self.n,self.n),dim=-1),dim=-3)
        
        # extract partial diagonals DA
        A = torch.diag_embed(torch.diagonal(A,dim1=-1,dim2=-3),dim1=-1,dim2=-3)
        
        # Calculate F*DA*F^T and return
        A = fft(ifft(A.reshape(m1,m2,self.n,self.n,self.n,self.n),dim=-1),dim=-3)
        
        return torch.real(A.reshape(m1,m2,self.n**2,self.n**2)).contiguous()

    def project_light(self,A):
          # To project the invarizing layers, we only need to take the mean in x-direction
          
          m1 = A.shape[0]
          m2 = A.shape[1]
          
          B = A.reshape(m1,m2,1,self.n,self.n).mean(-1,keepdim=True)
          B = B* torch.ones_like(A.reshape(m1,m2,1,self.n,self.n))
          return B.reshape(m1,m2,1,self.n**2)            

class OneDTranslationNet(TranslationNet):
    
    """
        class TranslationNet
            A Net for processing R^{n,n}- features in a manner 
            equivariant to translations in x-direction.
    """  
    
    def __init__(self,ns,n, device = 'cpu', fully_connected=False, ms = []):
        super(OneDTranslationNet,self).__init__(ns, n, device, fully_connected)
        
    
    def project(self,A):

        
        m1 = A.shape[0]
        m2 = A.shape[1]
        
        # calculate F^TAF, where F is Fourier operator in one direction
        A = ifft(fft(A.reshape(m1,m2,self.n,self.n,self.n,self.n),dim=-1),dim=-3)
        
        # extract partial diagonals DA
        A = torch.diag_embed(torch.diagonal(A,dim1=-1,dim2=-3),dim1=-1,dim2=-3)
        
        # Calculate F*DA*F^T and return
        A = fft(ifft(A.reshape(m1,m2,self.n,self.n,self.n,self.n),dim=-1),dim=-3)
        
        return torch.real(A.reshape(m1,m2,self.n**2,self.n**2)).contiguous()
    def project_light(self,A):
          # To project the invarizing layers, we only need to take the mean in x-direction
          
          m1 = A.shape[0]
          m2 = A.shape[1]
          
          B = A.reshape(m1,m2,1,self.n,self.n).mean(-1,keepdim=True)
          B = B* torch.ones_like(A.reshape(m1,m2,1,self.n,self.n))
          return B.reshape(m1,m2,1,self.n**2)


        


class RotationNet(TranslationNet):
    
    """
         class RotationNet
             A Net for processing R^{n,n}-features in a rotation (R_4) invariant manner.
            
        Note that the class extends Translation net, since many of its functionalities are very similar.
    """
    
    def __init__(self,ns,n, device = 'cpu',fully_connected=False, ms = []):
        
        # initialize a TranslationNet
        super(RotationNet,self).__init__(ns, n, device, fully_connected)
        
        # we do not need projection matrix here, so erase it
        self.projmat = None
        
        # last non-linearity 
        self.last_nonlin = torch.nn.LeakyReLU()
        
        if not fully_connected:
            self.bn = torch.nn.LayerNorm(2, device= self.device,elementwise_affine=False)  
            self.output_nonlin = torch.nn.Sigmoid()
        
        
    
    def forward(self,x):
        if not self.fully_connected:
            x = x.unsqueeze(1)
        x = super().forward(x) # first layers are handled by super class.
                                # this includes the fully connected layers if fully_connected
        if not self.fully_connected:
            # if not fully connected, need to handle final layer a bit differently
            x = torch.transpose(x,-1,-2)
            x = self.bn(x)
            x = torch.transpose(x,-1,-2)
            x = self.output_nonlin(x)

        
        
        return x
    
    
    def project(self,A):
        
        A = A.reshape((A.shape[0],A.shape[1],self.n,self.n,self.n,self.n))
        
        B = A.clone()
        
        C = B
        # Explicitly carry out the integration over the group.
        for k in range(3):   
            C = torch.flip(C,[-1])
            C = torch.flip(C,[-3])
            C = torch.transpose(C,-1,-2)
            C = torch.transpose(C,-3,-4)
            B += C

        return B.reshape((A.shape[0],A.shape[1],self.n**2,self.n**2))/4.0
    
    def reset(self):
            
        for k in range(len(self.layers)-1):
            self.layers[k].data = 1/(self.n*np.sqrt(self.ns[k]))*torch.randn(self.ns[k+1],self.ns[k],self.n**2,self.n**2).to(self.device)
           
            

        if self.fully_connected:
            self.layers[-1].data = 1/(np.sqrt(self.ms[0]))*(torch.randn(self.ms[0],self.ns[-1],1,1)*torch.ones(self.ms[0],self.ns[-1],1,self.n**2)).to(self.device)
            for k in range(1, len(self.ms)):
                lin = torch.nn.Parameter(torch.randn(self.ms[k-1],self.ms[k],device = self.device))
                lin.data = lin.data*self.n/np.sqrt(self.ms[k-1]*self.ms[k-1])
                self.fclayers.append(lin)
        else:
            self.layers[-1].data = 1/(self.n*np.sqrt(self.ns[k]))*torch.randn(self.ns[-1],self.ns[-2],self.n**2,self.n**2).to(self.device)
           
     
    def project_light(self,A):
         if self.fully_connected:
            A = A.reshape((A.shape[0],A.shape[1],1,1,self.n,self.n))
            
            B = A.clone()
            
            C = B
            # Explicitly carry out the integration over the group.
            for k in range(3):   
                C = torch.flip(C,[-1])
                C = torch.transpose(C,-1,-2)
                B += C

            return B.reshape((A.shape[0],A.shape[1],1,self.n**2))/4.0     
         else:
             # If we do not have fully connected things on top, we can project as usual
             return self.project(A)
    
class RotoTransNet(RotationNet):
    
    # a net both invariant to translations and rotations       

    # the representation is rho(rotation,translation)x = rho(rotation)rho(translation)x
    # this is not a representation of an abelian group, but it does the job!
    
    # accordingly, the projection operator to E is simply the projection onto the 'translation E' and then to the 'rotation-E'
    def project(self,A):
        
        # first project to the translation invariant filter, then project that to the rotation invariant.
        # this may look to simple, but is OK (compatibility condition holds for non-restricted filters)
        
        A = TranslationNet.project(self,A)
        A = super().project(A)
        return A
        
    def project_light(self,A):
        A = TranslationNet.project_light(self,A)
        A = super().project_light(A)
        return A
        


        