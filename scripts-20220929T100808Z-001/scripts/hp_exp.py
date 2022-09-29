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


# In[19]:


harmonic_sims = [ sim(g, config) for g in np.linspace(0, 10, 300)]
tr, te, va = make_dataset(harmonic_sims)


# In[20]:


gp = GPApproximation()
data = tr.sample(500)
gp.fit(data[['x', 'g']], data.psi)
print(f"### Trained GP in train dataset")



# In[21]:


def evaluate(gp, harmonic_sims, low=1, high=100, n=100):
    
    def _evaluate(g):
        gt = get_closest_sim(harmonic_sims, g)
        pr, pr_sigma = gp.predict(gt[['x', 'g']])
        return ((pr - gt.psi)**2).sum().mean()
    
    return np.array([_evaluate(g) for g in np.linspace(low, high, n)]).mean()


# In[ ]:


gp_hp_loss = evaluate(gp, harmonic_sims, low=10, high=15)
print(f"### Loss of the trained GP - {gp_hp_loss}")


# In[33]:


save_path = "HP_exp"
if os.path.isdir(save_path) == False:
    os.mkdir(save_path)
al_utils.plot([2, 4, 6], harmonic_sims, gp, save_file=os.path.join(save_path, f'{save_path}.svg'), act_op=False, min_gs=-5, max_gs=5)


# In[ ]:


print(f"###Starting active learning Process")
kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5], (1e-2, 1e2))
regressor = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
ids = [int(i) for i in np.linspace(0, len(tr)-1, 50)]
aclrnr = al_utils.ActLearn(
    regressor=regressor,
    init_ids=ids,
    trn_data=tr,
    save_path=save_path,
    compare_loss=gp_hp_loss,
    exp_name='Harmonic Potential'
)
aclrnr.train(harmonic_sims, low=10, high=15, gs=[2, 4, 6], min_gs=-5, max_gs=5)



# In[ ]:




