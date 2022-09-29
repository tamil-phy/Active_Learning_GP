#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os

from bec import *
from monet import *
import al_utils
from pathlib import Path


import warnings
warnings.filterwarnings('ignore')

np.random.seed(69)


# In[2]:


class OpticalLattice:
    config = get_optical_lattice_config()
    sims = [ sim(g, config) for g in np.linspace(0.1, 10, 100) ]
    tr, _, _ = make_dataset(sims)
   
def evaluate(gp, harmonic_sims, low=1, high=100, n=100):
    
    def _evaluate(g):
        gt = get_closest_sim(harmonic_sims, g)
        pr, pr_sigma = gp.predict(gt[['x', 'g']])
        return ((pr - gt.psi)**2).sum().mean()
    
    return np.array([_evaluate(g) for g in np.linspace(low, high, n)]).mean()


# In[3]:


gp = GPApproximation()
optical_lattice_data = OpticalLattice.tr.sample(500)
print(f"### Created Train, Test and Validation Sample")

gp.fit(optical_lattice_data[['x', 'g']], optical_lattice_data.psi)
OpticalLattice.gp = gp
print(f"### Trained GP in train dataset")



# In[4]:


Optical_loss_gp = evaluate(gp, OpticalLattice.sims, low=10, high=20)
print(f"### Loss of the trained GP - {Optical_loss_gp}")



# In[13]:


save_path = "OpticalLattice_exp"
if os.path.isdir(save_path) == False:
    os.mkdir(save_path)
al_utils.plot([7, 8, 9],OpticalLattice.sims, gp, save_file=os.path.join(save_path, f'{save_path}.svg'),act_op=False, min_gs=-10, max_gs=10)



# In[ ]:


print(f"###Starting active learning Process")
kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5], (1e-2, 1e2))
regressor = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
ids = [int(i) for i in np.linspace(0, len(OpticalLattice.tr)-1, 50)]
aclrnr = al_utils.ActLearn(
    regressor=regressor,
    init_ids=ids,
    trn_data=OpticalLattice.tr,
    save_path=save_path,
    compare_loss=Optical_loss_gp,
    exp_name='VTP_OpticalLattice'
)

aclrnr.train(OpticalLattice.sims, low=10, high=20,gs=[7, 8, 9], min_gs=-10, max_gs=10)



