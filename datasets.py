import torch
import numpy as np
import os
from torchvision.datasets import MNIST
from random import randint
import torchvision.transforms as transforms


class adMat(torch.utils.data.Dataset):
    
    
    """
      adMat 
      Dataset for handling data for graph connection problem.
      
      
    """
    def __init__(self,remake = False, aug=False, directory ='stoch_block',nbr_samples=10000,n=5,p=.5,q=.05, device = 'cpu', ):
        super(adMat,self)
        
        """
            If called without arguments, the data present in directory 'stoch_block' will be used
        
            remake - should the dataset be redrawn?
            aug - if True, the adjacence matrices are randomly permuted at load time
            nbr_samples - number of samples that should be used
            n - sizes of the graphs
            p,q - probabilities, only used when new data is generated
        """
        
        self.directory = directory
        self.nbr_samples = nbr_samples
        self.n = n
        
        self.aug = aug
        self.device = device

        if not os.path.isdir(directory):
                os.mkdir(directory)
                
        if remake:
            self.generate_dataset(directory,nbr_samples,n,p,q)
        
       
        
        

    def generate_dataset(self,directory, nbr_samples,n,p,q):
        
        string = os.path.join(directory,'graph')
        conn = 0 
        
    
        for k in range(nbr_samples):
            ad = np.zeros((2*n,2*n))
            # generate a graph according to the block Erdös-Renyi model
            for i in range(n):
                for j in range(n):
                    
                    
                    ad[i,n+j] = np.random.rand()<q # intercluster connection
                    if j>i:
                        ad[i,j] = np.random.rand()<p # intracluster connection
                        ad[i+n,j+n] = np.random.rand()<p
            
            ad = ad + ad.T #undirected graph

            # permute points to simulate randomly drawn subdivisions.
            ad = self.permuteAdj(ad) 
            
            d = np.diag(ad.sum(1))
            
            #check if directed by calculating second smallest eigenvalue of graph laplacian
            fied,_ = np.linalg.eigh(d-ad)
            fied = np.sort(fied)
            
            locstring = string + 'mat' +  str(k)+'.npy'
            labelstring = string + 'label' + str(k) + '.npy'
            label=1.0*(fied[1]>1e-15)
            conn +=label
            np.save(locstring,ad)
            np.save(labelstring,label)
            
        # print the number of graphs that are connected
        # (if the dataset is very biased towards either class, one may want to rechoose the parameters)
        print("New graph dataset generated.")
        print("Fraction of connected graphs: "+str(conn))
    
    def __getitem__(self,index):
        
        gstring = os.path.join(self.directory,'graphmat'+str(index)+'.npy')
        lstring = os.path.join(self.directory,'graphlabel'+str(index)+'.npy')
        
        graph = np.load(gstring)
        label = np.load(lstring)
        
        if self.aug:
            graph = self.permuteAdj(graph)
        
        return torch.tensor(graph.reshape(-1,4*self.n**2),dtype=torch.float,device= self.device),torch.tensor(label,dtype=torch.float,device = self.device)
        
    def __len__(self):
        
        
        
        return self.nbr_samples
        
    
    
    @staticmethod
    def permutevec(v):
        # permutes (using a single permutation over batch) a vector
        # v is of size batch,n
        n = v.size(-1)
        
        perm = torch.nn.randperm(n)
        
        return v[:,perm]
    
    def permuteAdj(self,A):
        # permutes (using a single permutation over batch) a matrix
        # If A is torch.tensor,  A is assumed to be of size batch,n,n
        # otherwise, it is of size (n,n)
        try:
            n = 2*self.n
            shp = A.shape
            A = A.reshape(-1,n,n)
            perm = torch.randperm(n)
        
            A = A[:,:,perm]
            A = A[:,perm,:]
            return A.reshape(shp)
        except(TypeError):
            n = A.shape
            n = n[-1]
        
            perm = np.random.permutation(n)
        
            A = A[:,perm]
            A = A[perm,:]
            return A

class SimpleShapes(torch.utils.data.Dataset):
    
    
    """
        A class for handling simple shape segmentation task
    """
    
    def __init__(self, n=14,remake = False, nbr_samples=10000, directory = 'simpshape', device = 'cpu'):
        
        """
            If called without arguments, the data will be loaded from the directory 'simp-shape
            
                n - size of images is nxn.
                remake - should the dataset be redrawn?
                nbr_samples - number of samples that should be used
                n - sizes of the graphs
                p,q - probabilities, only used when new data is generated
        """
        self.nbr_samples = nbr_samples
        self.n = n
        self.directory = directory
        self.device = device
        
        if not os.path.isdir(directory):
            os.mkdir(directory)
     
            self.generate_dataset()
        
        
    def __len__(self):
        return self.nbr_samples
    
    def __getitem__(self,index):
        
        ret = []
        names = ['img','trig','pent']
        for name in names:
            string = os.path.join(self.directory,'shps'+name+str(index)+'.npy')
            ding= torch.from_numpy(np.load(string).flatten())
            ding = ding.float()
            #ding= ding.to(self.device)
            ret.append(ding.reshape(1,-1))
        
       
        
        return ret[0],(ret[1],ret[2])
    
    
    def generate_dataset(self):
        import matplotlib.pyplot as plt
        from cv2 import resize
        
        # The two shapes
        
        omega = 2*np.pi/5
        pentagon = np.zeros((6,2))
        for k in range(6):
            pentagon[k,0] = np.cos(omega*k)
            pentagon[k,1] = np.sin(omega*k)
            
        triangle = np.zeros((4,2))
        omega= 2*np.pi/3
        for k in range(4):
            triangle[k,0] = np.cos(omega*k)
            triangle[k,1] = np.sin(omega*k)
        
        
        # make figures for image and masks
        wf = plt.figure(0)
        tf = plt.figure(1)
        pf = plt.figure(2)
        whole = wf.subplots()
        trig = tf.subplots()
        pent = pf.subplots()
        
        
        for ind in range(self.nbr_samples):
            
            string = os.path.join(self.directory,'shps')
            
            for figg in [whole,trig,pent]:
                figg.cla()
                figg.set_aspect('equal')
                figg.axis('off')
                figg.axis((-1,1,-1,1))
            mask = whole    
            for k in range(1):
                # choose either triangle or pentagon with equal probability
                if np.random.rand()>.5:
                    shape = triangle    
                    mask = trig
                else:
                    shape = pentagon
                    mask = pent
                
                # add the shape to the canvas
                r = (.7+.1*np.random.rand())
                v = (.5-1*np.random.rand(1,2))
                patch1= plt.Polygon(r*shape + v,fill=True)
                patch2= plt.Polygon(r*shape + v,fill=True)
                whole.add_patch(patch1)
                mask.add_patch(patch2)
            
            
            
            for f in [wf,tf,pf]:
                f.tight_layout()
                f.canvas.draw()
            
           
           
           
            img = self.to_numpy(wf)
            tmsk = self.to_numpy(tf)
            pmsk = self.to_numpy(pf)
        
            
            # resize the images using opencv:s resize function.
            r = resize(img,(self.n,self.n),anti_aliasing=True)
            t = resize(tmsk,(self.n,self.n),anti_aliasing=True)
            p = resize(pmsk,(self.n,self.n),anti_aliasing=True)
            
            
            # turn the segmentation masks into 0-1 images.
            t = t.mean(-1)
            m = t.min()
            M = t.max()
            if not m==M:
                t = (t-M)/(m-M)
            else:
                t = t/M
            t = (t>.5)*1.0
            
            p = p.mean(-1)
            m = p.min()
            M = p.max()
            if not m==M:
                p = (p-M)/(m-M)
            else:
                p = p/M
        
            p = (p>.5)*1.0
            
            rstring = string + 'img' +  str(ind)+'.npy'
            tstring = string + 'trig' + str(ind) + '.npy'
            pstring = string + 'pent' + str(ind)
          
            np.save(rstring,r.mean(-1))
            np.save(tstring,t) 
            np.save(pstring,p)
        
       
        
    def to_numpy(self,fig):
        img = fig.canvas.tostring_rgb()
        img = np.frombuffer(img,dtype=np.uint8)
    
        return img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
def identity(x):
    
    # 'dummy function' to use as transformation function when the group acts trivially
    return x

class Shift(torch.nn.Module):
    
    """
        A class for randomly shifting images. For NxN -images, set n parameter equal to N 
    """
    
    def __init__(self,n):
        super(Shift,self).__init__()
        self.n = n
    
    def forward(self,x):
        
        s1 = randint(0,self.n)
        s2 = randint(0,self.n)
        return torch.roll(x,dims=(-2,-1),shifts=(s1,s2))
    
class Xshift(torch.nn.Module):
        
    """
        A class for randomly shifting images in x-directoin. For NxN -images, set n parameter equal to N 
    """
    
    def __init__(self,n):
        super(Xshift,self).__init__()
        self.n = n
    
    def forward(self,x):
        
        s = randint(0,self.n)
        return torch.roll(x,dims=(-1),shifts=(s))
    
    

class Xshift(torch.nn.Module):
        
    """
        A class for randomly shifting images in x-directoin. For NxN -images, set n parameter equal to N 
    """
    
    def __init__(self,n):
        super(Xshift,self).__init__()
        self.n = n
    
    def forward(self,x):
        
        s = randint(0,self.n)
        return torch.roll(x,dims=(-1),shifts=(s))

class Rotate(torch.nn.Module):
    
    """
        A class for randomly rotating images. For NxN -images, set n parameter equal to N 
        
        Can only be used together with other code. It stores an internal index making sure that
        image masks are rotated in the correct manenr
    """
    
    def __init__(self,n, new = False):
        super(Rotate,self).__init__()
        self.n = n
        self.idx = 0
        self.r = 1
        self.new = new # to globally override storing mechanism
    def forward(self,x, new = False):
        
        if self.new:
<<<<<<< HEAD
            new=self.new        # discretely rotates image. 
=======
            new=self.new
        # discretely rotates image. 
>>>>>>> refs/remotes/origin/main
        # assumes that x is in form [batch,dim,n**2]
     
        
        b = x.shape[0]        
        x.reshape(x.shape[0],x.shape[1],self.n,self.n)

        if new: # set new = True to override storing mechanism
            self.r = randint(0,4)
            self.idx= -1
        elif self.idx == 0: # if idx == 0, it is time to draw a new index
            self.r = randint(0,4)
            
        self.idx = (self.idx + 1)%3 # in this way, a new rotation is drawn only every third application of the class
        r = self.r
            
        # apply one of the four rotations in the class
        if r == 0:
            pass
            
        if r == 1:
            x = x.flip(-1)
            x = x.transpose(-1,-2)
        
        if r == 2:
            x = x.flip([-1,-2])
            
        if r == 3:
            x = x.transpose(-1,-2)
            x = x.flip(-2)
        
        return x.reshape(b,-1,self.n**2)
<<<<<<< HEAD
=======
    

>>>>>>> refs/remotes/origin/main
class RotoShift(torch.nn.Module):
    
    """
        A class for simultaneous random rotation and translations images. For NxN -images, set n parameter equal to N 
        
        Do not use for masks, will generate a new rotoshift each time
    """
    
    def __init__(self, n):
        super(RotoShift,self).__init__()
        self.shift = Shift(n)
        self.rot = Rotate(n,True)
    
    def forward(self,x):
        x = self.shift(x)
        return self.rot(x)
            
def get_MNIST_loader(batch_size,n=14, root = 'mnist'):
    
    """
        wrapper for getting a resized version of mnist.
        Will download mnist if not present in specified directory.
        
    """
    mnist = MNIST(root=root,train=False, download=True, transform =
                  transforms.Compose([transforms.Resize((n,n)),transforms.ToTensor()]))#,Shift(14)]))

    loader = torch.utils.data.DataLoader(mnist,batch_size=batch_size)
    
    return loader