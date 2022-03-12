#!/usr/bin/env python
# coding: utf-8

# ## Toy example: Inferring the mean of Gaussians
# 
# #### comparing the multi-round SNPE approach against our new incremental approach.
# 
# Goal of this little toy example is to show that provided our parameters are independent of each other, we need less simulations to derive a good approximation of our parameters.

# In[1]:


import sys
sys.path.append('../code/')

import utils
from utils.helpers import get_time
from utils import inference

from utils.sbi_modulated_functions import Combined


from utils.helpers import get_time

from utils.simulation_wrapper import SimulationWrapper


# sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.inference import SNPE_C

import sbi


# Defining a function (Gaussian) that takes a arbitrary number of thetas (parameters for a Gaussian mean) and samples from a Gaussian with this mean and a standard deviation of 0.1

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import torch


def Gaussian(thetas, normal_noise=0.1):
    
    np.random.seed(np.random.choice(1000))
    
    gauss_list = []
    
    for theta in thetas:
    
        mu, sigma = theta, normal_noise # mean and standard deviation

        s = np.random.normal(mu, sigma, 1)
    
        
        gauss_list.append(s[0])
        
    gauss_obs = torch.tensor(gauss_list)
    
    return gauss_obs
    




true_thetas = torch.tensor([[3.0, 6.0, 20.0, 10.0, 90.0, 55.0, 27.0, 27.0, 4.0, 70.0, 5.0, 66.0, 99.0, 40.0, 45.0]])
parameter_names = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15']

prior_max = [100.0] * 15
prior_min = [1.0] * 15


# In[6]:


num_simulations_list = [600, 800, 1000]


# In[7]:



list_collection = torch.load('list_collection.pt')


# In[9]:


range_list = [5,10, 15]



list_collection_inc = torch.load('list_collection_inc.pt')




def calc_KL_highd(posterior):
    
    sample = posterior.sample((10000,))
    
    analytic = torch.distributions.normal.Normal(true_thetas, 0.1)
    
    analytic_sample = analytic.sample((10000,))
    
    out = F.kl_div(analytic_sample, sample)
    
    return out


# In[ ]:


def calc_KL_1d(posterior):
    
    sample = posterior.sample((10000,))
    
    analytic = torch.distributions.normal.Normal(true_thetas, 0.1)
    
    analytic_sample = analytic.sample((10000,)).squeeze(1)
    
    out_list = []
    for i in range(len(posterior)):
        
        out = F.kl_div(analytic_sample[:,i], sample[:,i])
        out_list.append(out)
    
    return out_list


# In[11]:


def KL_Gauss(X, Y):
    
    sample_x = X.sample((1000,))
    mu_x = torch.mean(sample_x, dim=0)
    var_x = torch.std(sample_x, dim=0)

    var_y = Y.stddev

    mu_y = Y.mean
    
    
    return torch.mean(np.log(var_y/var_x) + (var_x**2 + (mu_x - mu_y)**2)/(2*var_y**2) -(1/2))


def calc_KL_1d(X, Y):
    
    sample_x = X.sample((1000,))
    mu_x = torch.mean(sample_x, dim=0)
    var_x = torch.std(sample_x, dim=0)
    
    print(var_x)
    print(mu_x)


    var_y = Y.stddev

    mu_y = Y.mean
    
    print(mu_y)
    print(var_y)
    
    print(np.log(var_y/var_x) + (var_x**2 + (mu_x - mu_y)**2)/(2*var_y**2) -(1/2))
    
    return np.log(var_y/var_x) + (var_x**2 + (mu_x - mu_y)**2)/(2*var_y**2) -(1/2)




# In[77]:



# In[ ]:


import torch

analytic = torch.distributions.normal.Normal(true_thetas, 0.1)

analytic.stddev


# In[12]:





analytic = torch.distributions.normal.Normal(true_thetas, 0.1)


overall_snpe_list = []


## for round
for posterior_snpe_list in list_collection:
    
    KL_snpe = []
    KL_snpe_1d = []
    
    
    ## for number of simulations
    for posterior_snpe in posterior_snpe_list:


        #KL = KLdivergence(posterior_snpe, sample_y)
        KL = KL_Gauss(posterior_snpe, analytic)


        KL_1d = calc_KL_1d(posterior_snpe, analytic)

        KL_snpe_1d.append(KL_1d)

        #KL_snpe_sum.append(sum_KL)

        KL_snpe.append(KL)
        
    overall_snpe_list.append(KL_snpe)


    
    


# In[51]:


obs_real = Gaussian(true_thetas[0, 0:])



analytic = torch.distributions.normal.Normal(true_thetas, 0.1)


overall_incremental_list = []

for posterior_incremental_list in list_collection_inc:
    
    KL_incremental = []

    for posterior_incremental in posterior_incremental_list:

        posterior_incremental.set_default_x(obs_real)

        #KL = KLdivergence(posterior_incremental, sample_y)

        KL = KL_Gauss(posterior_incremental, analytic)

        #KL_1d = calc_KL_1d(posterior_incremental, analytic)

        #KL_incremental_1d.append(KL_1d)


        KL_incremental.append(KL)

        
    overall_incremental_list.append(KL_incremental)

    


# In[37]:


len(overall_incremental_list)


# In[65]:


mean_incremental = np.mean(np.array(overall_incremental_list), axis=0)

print(mean_incremental)

stdev_incremental = np.std(np.array(overall_incremental_list), axis=0)

print(stdev_incremental)


lower_incremental = mean_incremental - [element * 1.96 for element in stdev_incremental]

upper_incremental = mean_incremental + [element * 1.96 for element in stdev_incremental]


# In[66]:


mean_snpe = np.mean(np.array(overall_snpe_list), axis=0)

print(mean_snpe)

stdev_snpe = np.std(np.array(overall_snpe_list), axis=0)

print(stdev_snpe)


lower_snpe = mean_snpe - [element * 1.96 for element in stdev_snpe]

upper_snpe = mean_snpe + [element * 1.96 for element in stdev_snpe]


# ### Compare KL-divergence of snpe approach with incremental approach in a plot:
# 
# #### x = number of simulations per round/step

# In[67]:


figure_mosaic = """
ACC
BCC
"""

fig, axes = plt.subplot_mosaic(layout=figure_mosaic, figsize=(11, 8))

    

axes['B'].plot(num_simulations_list, mean_incremental, '-o', color='blue')
axes['A'].plot(num_simulations_list, mean_snpe, '-o',  color='orange')

axes['B'].plot(num_simulations_list, upper_incremental, '--', color='blue')
axes['A'].plot(num_simulations_list, upper_snpe, '--',  color='orange')

axes['B'].plot(num_simulations_list, lower_incremental, '--', color='blue')
axes['A'].plot(num_simulations_list, lower_snpe, '--',  color='orange')


axes['C'].plot(num_simulations_list, mean_incremental, '-o',label='incremental', color='blue')
axes['C'].plot(num_simulations_list, mean_snpe, '-o', label='snpe', color='orange')

axes['C'].plot(num_simulations_list, upper_incremental, '--', color='blue')
axes['C'].plot(num_simulations_list, upper_snpe, '--',  color='orange')

axes['C'].plot(num_simulations_list, lower_incremental, '--',  color='blue')
axes['C'].plot(num_simulations_list, lower_snpe, '--',  color='orange')


axes['C'].fill_between(x= num_simulations_list, y1=lower_incremental, y2=upper_incremental, color='blue', alpha=0.2)
axes['C'].fill_between(x= num_simulations_list, y1=lower_snpe, y2=upper_snpe, color='orange', alpha=0.2)


axes['B'].fill_between(x= num_simulations_list, y1=lower_incremental, y2=upper_incremental, color='blue', alpha=0.2)
axes['A'].fill_between(x= num_simulations_list, y1=lower_snpe, y2=upper_snpe, color='orange', alpha=0.2)


axes['B'].fill_between(x= num_simulations_list, y1=lower_incremental, y2=upper_incremental, color='blue', alpha=0.2)
axes['A'].fill_between(x= num_simulations_list, y1=lower_snpe, y2=upper_snpe, color='orange', alpha=0.2)


#plt.title('KL loss')
axes['A'].legend()
axes['B'].legend()
axes['C'].legend()

plt.xlabel('simulations per round')
plt.ylabel('KL divergence')

axes['A'].set_title('SNPE')
axes['B'].set_title('Incremental')


#axes['B'].set_xticklabels(['0k','2k', '4k', '6k', '8k', '10k'])
#axes['A'].set_xticklabels(['0k','2k', '4k', '6k', '8k', '10k'])
#axes['C'].set_xticklabels(['0k','2k', '4k', '6k', '8k', '10k'])
#plt.xticks(['1k', '3k', '5k', '10k'])


# In[44]:

plt.savefig('Gauss_plot.png')


