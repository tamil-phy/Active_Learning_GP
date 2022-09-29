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


class DoubleWell:
    config = get_double_well_config()
    sims = [ sim(g, config) for g in np.linspace(0.1, 5, 100) ]
    tr, _, _ = make_dataset(sims)
   
def evaluate(gp, harmonic_sims, low=1, high=100, n=100):
    
    def _evaluate(g):
        gt = get_closest_sim(harmonic_sims, g)
        pr, pr_sigma = gp.predict(gt[['x', 'g']])
        return ((pr - gt.psi)**2).sum().mean()
    
    return np.array([_evaluate(g) for g in np.linspace(low, high, n)]).mean()


# In[3]:


gp = GPApproximation()
doublewell_data = DoubleWell.tr.sample(500)
print(f"### Created Train, Test and Validation Sample")
gp.fit(doublewell_data[['x', 'g']], doublewell_data.psi)
DoubleWell.gp = gp
print(f"### Trained GP in train dataset")



# In[23]:


double_loss_gp_loss = evaluate(gp, DoubleWell.sims, low=5, high=10)
print(f"### Loss of the trained GP - {double_loss_gp_loss}")


# In[24]:


save_path = "DoubleWell_exp"
if os.path.isdir(save_path) == False:
    os.mkdir(save_path)
al_utils.plot([1, 20, 30], DoubleWell.sims, gp, save_file=os.path.join(save_path, f'{save_path}.svg'), act_op=False, min_gs=-5, max_gs=5)


# In[20]:


print(f"###Starting active learning Process")
kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5], (1e-2, 1e2))
regressor = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
ids = [int(i) for i in np.linspace(0, len(DoubleWell.tr)-1, 50)]
aclrnr = al_utils.ActLearn(
    regressor=regressor,
    init_ids=ids,
    trn_data=DoubleWell.tr,
    save_path=save_path,
    compare_loss=double_loss_gp_loss,
    exp_name='VTP_DoubleWell'
)

aclrnr.train(DoubleWell.sims, low=5, high=10, gs=[1, 20, 30], min_gs=-5, max_gs=5)



# In[ ]:




