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


# In[35]:


def kdv_exact(x,t,k1,c):
    return 0.5*k1**2 / (np.cosh(0.5*(k1 * x - k1**3 * t + c)**2))

x = np.linspace(-20.0, 20.0, 512)
t = np.linspace(0, 50., 300)

X, T = np.meshgrid(x, t)

k1 = 0.5
c = 1

ans = np.array(kdv_exact(np.ravel(X), np.ravel(T), k1, c))
sol = ans.reshape(X.shape)

def data_kdv(x,t,sol):
    df = pd.DataFrame({
    'x' : x,
    'g' : t,
    'psi' : sol
    })
    return df

sample = [ data_kdv(X[i,:],T[i,:],sol[i,:]) for i in range(0,300)]
tr, te, va = make_dataset(sample)
print(f"### Created Train, Test and Validation Sample")


# In[36]:


gp = GPApproximation()
data = tr.sample(500)
gp.fit(data[['x', 'g']], data.psi)
print(f"### Trained GP in train dataset")


# In[37]:


def evaluate(gp, sample, low=1, high=100, n=100):
    
    def _evaluate(g):
        gt = get_closest_sim(sample, g)
        pr, pr_sigma = gp.predict(gt[['x', 'g']])
        return ((pr - gt.psi)**2).sum().mean()
    
    return np.array([_evaluate(g) for g in np.linspace(low, high, n)]).mean()


# In[38]:


gp_kdv_loss = evaluate(gp, sample, low=50, n=100, high=55)
print(f"### Loss of the trained GP - {gp_kdv_loss}")


# In[53]:


from importlib import reload
reload(al_utils)


# In[54]:


save_path = "KDV_EXP"
if os.path.isdir(save_path) == False:
    os.mkdir(save_path)
al_utils.plot([5, 25, 52], sample, gp, save_file=os.path.join(save_path, f'{save_path}.svg'), act_op=False, min_gs=-20, max_gs=20)



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
    compare_loss=gp_kdv_loss,
    exp_name='KDV'
)

aclrnr.train(sample, low=50, n=100, high=55, gs=[5, 25, 52], min_gs=-20, max_gs=20)



# In[ ]:




