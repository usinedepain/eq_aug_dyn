import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
import sys


"""
    Plots for producing plots of the results, as generated by the training functions.
    
    Call with argument 'perm', 'trans' or 'rot' depending on which results you want to plot
    Second argument is number of experiments performed
"""

def get_perm_statistics(dir_name,nbr_exp,tp):
    
    dirr_name = os.path.join('results',dir_name)
    filename = os.path.join(dirr_name + '_00','eq')
    
    f = open(filename,'r')
    N = len(f.readlines())
    f.close()
    
    dirr_name = dir_name
    
    varnames = 'loss','lay0','lay1','shunt0','shunt1'
    res={}
    for var in varnames:
        res[var] = np.zeros((N+1,nbr_exp)) 
    
    dir_name = os.path.join('results', dirr_name)
    

    for n in range(nbr_exp):
        file = os.path.join(dir_name+'_0'+str(n),tp)

        f = open(file)    
        for k,line in enumerate(f.readlines()):
                g=line.split()
                
                res['loss'][k+1,n] = float(g[1].replace(':',''))
                res['lay0'][k+1,n] = float(g[3].replace('[',''))
                res['lay1'][k+1,n] = float(g[4].replace(']',''))
                res['shunt0'][k+1,n] = float(g[6].replace('[','').replace(',',''))
                res['shunt1'][k,n] = float(g[7].replace(']',''))
        f.close()
            
        
        
        
        
    return res


def get_translation_statistics(dir_name,nbr_exp,tp):
    
    varnames = 'loss','lay0','lay1','lay2','shunt0','shunt1', 'shunt2'
    
    dirr_name = os.path.join('results',dir_name)
    filename = os.path.join(dirr_name + '_00','eq')
    
    f = open(filename,'r')
    N = len(f.readlines())
    f.close()
    res={}
    for var in varnames:
        res[var] = np.zeros((N+1,nbr_exp)) 
        

    for n in range(nbr_exp):
        file = os.path.join(dirr_name+'_0'+str(n),tp)

        f = open(file)    
        for k,line in enumerate(f.readlines()):
                    
               g=line.split()
               res['loss'][k+1,n] = float(g[1].replace(':',''))
               res['lay0'][k+1,n] = float(g[3].replace('[',''))
               res['lay1'][k+1,n]  = float(g[4].replace(',',''))
               res['lay2'][k+1,n]  = float(g[5].replace(']',''))
               res['shunt0'][k+1,n]  = float(g[7].replace('[','').replace(',',''))
               res['shunt1'][k+1,n]  = float(g[8].replace(',',''))
               res['shunt2'][k+1,n]  = float(g[9].replace(']',''))
        f.close()
        
    return res



def get_rotation_statistics(dir_name,nbr_exp,tp):
    
   dirr_name = os.path.join('results',dir_name)
   filename = os.path.join(dirr_name + '_01','eq')
    
   f = open(filename,'r')
   N = len(f.readlines())
   f.close()
    
   varnames = 'loss','lay0','lay1','lay2','lay3','shunt0','shunt1', 'shunt2','shunt3'
   res={}
   for var in varnames:
       res[var] = np.zeros((N+1,nbr_exp)) 
       res[var][:,0] =0.0
        
    
    # 1 3 4 6 7 9 10
   for n in range(nbr_exp):
        file = os.path.join(dirr_name+'_0'+str(n),tp)
        f = open(file)    
        for l,line in enumerate(f.readlines()):
            g=line.split()
           
            k = l+1
            
            
            res['loss'][k,n] = float(g[1].replace(':',''))
            res['lay0'][k,n] = float(g[3].replace('[',''))
            res['lay1'][k,n] = float(g[4].replace(',',''))
            res['lay2'][k,n]= float(g[5].replace(']',''))
            res['lay3'][k,n] = float(g[6].replace(']',''))
            res['shunt0'][k,n]= float(g[8].replace('[','').replace(',',''))
            res['shunt1'][k,n] = float(g[9].replace(',',''))
            res['shunt2'][k,n] = float(g[10].replace(']','').replace(',',''))
            res['shunt3'][k,n] = float(g[11].replace(']',''))
           

            f.close()
        
   return res

def plot_stat_to_stat(eqres,augres,non_augres,stat0,stat1,name = 'Unknown', y2 = 10e-6, non_aug=True, x= 2*10**-2, y1 = 2*10**-4):

    # plot two statistics against each other. Produces a plot similar (slightly less tidy) to the one in the paper
    

    fig = plt.figure(name)
    ax0 = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax = [ax1,ax2]
# Turn off axis lines and ticks of the big subplot
    ax0.spines['top'].set_color('none')
    ax0.spines['bottom'].set_color('none')
    ax0.spines['left'].set_color('none')
    ax0.spines['right'].set_color('none')
    ax0.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    for k in range(2):
       
        ax[k].ticklabel_format(axis='both',scilimits=(0,0))
        ax[k].plot(eqres[stat0], eqres[stat1],c='green', alpha=.1,linestyle ='solid')
        ax[k].plot(augres[stat0], augres[stat1],c='blue', alpha=.1, linestyle = 'dashed')
        if non_aug:
            ax[k].plot(non_augres[stat0], non_augres[stat1],c='red', alpha=.1,linestyle ='dotted')
        ax[k].plot(np.median(eqres[stat0],1), np.median(eqres[stat1],1),c='green', alpha=1, linestyle = 'solid', label = 'Equivariant')
        ax[k].plot(np.median(augres[stat0],1), np.median(augres[stat1],1),c='blue', alpha=1, linestyle = 'dashed', label = 'Augmented')
        if non_aug:
            ax[k].plot(np.median(non_augres[stat0],1), np.median(non_augres[stat1],1),c='red', alpha=1, linestyle = 'dotted', label = 'Normal')
            
    
    ax2.set_ylim([-y2/10,y2])
    # remove this later
    ax2.set_xlim([0,x])
    ax2.set_ylim([-y2/10,y2])
    ax1.set_xlim([0,x])
    ax1.set_ylim([-y1/10,y1])
    
    
    
    ax0.set_xlabel('$\Vert A - A^0\Vert$')
    ax0.set_ylabel('$\Vert A_{\mathcal{E^\perp}}\Vert$')
    ax0.set_title(name.replace('_',' '))
    #ax2.legend()
    plt.subplots_adjust(left=.1)
    interactive('false')


    

    
    
def collect_layer_stats(eqres,augres,non_augres,nlay):
    reses = [eqres,augres,non_augres]
    
    # sum up the total errors /shifts
    
    for res in reses:
        res['totlay'] = 0.0
        res['totshunt'] = 0.0
        
        for k in range(nlay):
            res['totlay'] += res['lay'+str(k)]**2
            res['totshunt'] += res['shunt'+str(k)]**2
        
        res['totlay']=res['totlay']**.5
        res['totshunt']= res['totshunt']**.5
    
    return eqres,augres,non_augres


def plot_perm_statistics(n, x= 2*10**-2, y1 = 2*10**-4, y2 = 2**10**-6):
    eqres = get_perm_statistics('res_perm',n,'eq')
    augres = get_perm_statistics('res_perm',n,'aug')
    non_augres = get_perm_statistics('res_perm',n,'nonaug')
    
    
    eqres,augres,non_augres = collect_layer_stats(eqres,augres,non_augres, 2)
    
    x,y1,y2 = calclims(eqres,augres,non_augres)
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 14
    
    plot_stat_to_stat(eqres,augres,non_augres,'totshunt','totlay','Permutations',x = x, y1 = y1, y2 = y2)
    plt.show()
    
def plot_transl_statistics(n,addon='',x= 2*10**-2, y1 = 2*10**-4, y2 = 2**10**-6, notall = True, name = 'Translations'):
    if 'rot' in addon:
        eqres = get_translation_statistics('res_' + addon,n,'eq')
        augres = get_translation_statistics('res_'+addon,n,'aug')
        non_augres = get_translation_statistics('res_'+addon,n,'nonaug')
        name = addon
    else:
        name = 'trans' + addon
        eqres = get_translation_statistics('res_trans'+addon,n,'eq')
        augres = get_translation_statistics('res_trans'+addon,n,'aug')
        non_augres = get_translation_statistics('res_trans'+addon,n,'nonaug')
    
    eqres,augres,non_augres = collect_layer_stats(eqres,augres,non_augres, 3)
    
    if notall:
        x,y1,y2 = calclims(eqres,augres,non_augres)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 14
    
    
    plot_stat_to_stat(eqres,augres,non_augres,'totshunt','totlay',name,y2=y2, x=x, y1=y1)
    #plt.savefig(os.path.join('plots',name+'_rerun'+'.svg'))
    #plt.show()
    
def plot_rotation_statistics(n):
    eqres = get_rotation_statistics('res_rot',n,'eq')
    augres = get_rotation_statistics('res_rot',n,'aug')
    non_augres = get_rotation_statistics('res_rot',n,'nonaug')
    
    eqres,augres,non_augres = collect_layer_stats(eqres,augres,non_augres, 4)
    
    x,y1,y2 = calclims(eqres,augres,non_augres)
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 14
    
    x,y1,y2 = calclims(eqres,augres,non_augres)
    
      
    plot_stat_to_stat(eqres,augres,non_augres,'totshunt','totlay','Rotations',x = x, y1 = y1, y2 = y2)
    plt.title('Rotation')
    plt.show()

def calclims(eqres,augres,non_augres):
    return (np.median(eqres['totshunt'],1).max()*1.5,  np.median(non_augres['totlay'],1).max()*1.5, np.median(augres['totlay'],1).max()*1.5)
    
def plot_all_translations(n):
    
    # plot statistics for experiments with different groups
    
    # determine limits
    eqres = get_translation_statistics('res_trans',n,'eq')
    augres = get_translation_statistics('res_trans',n,'aug')
    non_augres = get_translation_statistics('res_trans',n,'nonaug')
    
    eqres,augres,non_augres = collect_layer_stats(eqres,augres,non_augres, 3)
    (x0,y10,y20) = calclims(eqres,augres,non_augres)
    
    
    # plot statistics one after the other
    
    exps = ['rot_class','rot_trans','trans','_one']
    colors = ['red','green','blue','black']    
    for k in range(4):
        exp = exps[k]
        c = colors[k]
        if exp == '_one':
            eqres = get_translation_statistics('res_trans_one',n,'eq')
            augres = get_translation_statistics('res_trans_one',n,'aug')
            non_augres = get_translation_statistics('res_trans_one',n,'nonaug')
        else:
            eqres = get_translation_statistics('res_'+exp,n,'eq')
            augres = get_translation_statistics('res_'+exp,n,'aug')
            non_augres = get_translation_statistics('res_'+exp,n,'nonaug')
        
        eqres,augres,non_augres = collect_layer_stats(eqres,augres,non_augres, 3)
        x,y1,y2 = calclims(eqres,augres,non_augres)
        
        if exp == 'trans':
            exp = ''
        plot_transl_statistics(n,exp,x=x,y1=y1, y2 = y20*y1/y10, notall=False, name = exp)
    plt.show()

    
    
        
    
    
if __name__ == "__main__":
    
    name = sys.argv[1]
    nbr_exp = int(sys.argv[2])
    
    if name == 'all_t':
        plot_all_translations(nbr_exp)
    
    
    

    if 'perm' in name:
        plot_perm_statistics(nbr_exp)
    if 'rot' in name:
        if 'class' in name or 'trans' in name:
            plot_transl_statistics(nbr_exp,name)
        else:
            plot_rotation_statistics(nbr_exp)
    elif 'trans' in name:
        if 'one' in name:
            plot_transl_statistics(nbr_exp,'_one')
        else:
            plot_transl_statistics(nbr_exp)
    
    
    