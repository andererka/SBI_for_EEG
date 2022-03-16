from utils.simulation_wrapper import SimulationWrapper
from data_load_writer import load_from_file as lf
from data_load_writer import write_to_file


from summary_features.calculate_summary_features import (
    calculate_summary_stats_temporal, calculate_summary_statistics_alternative

)

import numpy as np
import torch
import json
import pandas as pd
import seaborn as sns

import shutil

import datetime

from utils.helpers import get_time

from utils.sbi_modulated_functions import Combined

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn

from utils import inference


import pickle
import sys

import os

def Gaussian(thetas, normal_noise=1):
    
    np.random.seed(np.random.choice(1000))
    
    gauss_list = []
    
    for theta in thetas:
    
        mu, sigma = theta, normal_noise # mean and standard deviation

        s = np.random.normal(mu, sigma, 1)
    
        
        gauss_list.append(s[0])
        
    gauss_obs = torch.tensor(gauss_list)
    
    return gauss_obs



def main(argv):


    try:
        density_estimator = argv[0]
    except:
        density_estimator = 'mdn'
    try:
        num_workers = int(argv[1])
    except:
        num_workers = 8
    try:
        slurm = bool(int(argv[2]))
    except:
        slurm = True
    try:
        experiment_name = argv[3]
    except:
        experiment_name = "ERP_sequential"


    file_writer = write_to_file.WriteToFile(
    experiment=experiment_name,
    density_estimator=density_estimator,
    slurm=slurm,
    )

    try:
        os.mkdir(file_writer.folder)
    except:
        print('file exists')

    os.chdir(file_writer.folder)

        


    # ### Larger comparison with KL-divergence between analytic and inferred posterior

    # ### Calculate posterior for different number of simulations: 1k,  3k, 5k, 10k

    # ### starting with multi-round snpe

    # In[5]:

    import torch


    true_thetas = torch.tensor([[3.0, 6.0, 20.0, 10.0, 90.0, 55.0, 27.0, 27.0, 4.0, 70.0, 5.0, 66.0, 99.0, 40.0, 45.0]])
    parameter_names = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15']

    prior_max = [100.0] * 15
    prior_min = [1.0] * 15


    # In[6]:


    num_simulations_list = [600, 800, 1000, 2000, 3000]


    # In[7]:


    obs_real = Gaussian(true_thetas[0])

    obs_real


    # In[ ]:


    list_collection = []

    obs_real = Gaussian(true_thetas[0])

    start_time = datetime.datetime.now()

    for i in range(10):
        

        posterior_snpe_list = []

        for num_simulations in num_simulations_list:
            
            prior = utils.torchutils.BoxUniform(low=prior_min, high = prior_max)
            simulator_stats, prior = prepare_for_sbi(Gaussian, prior)
            
            inf = SNPE(prior, density_estimator=density_estimator)
            
            proposal = prior

            for j in range(3):

                print('round: ', j)
                
                theta, x = simulate_for_sbi(
                    simulator_stats,
                    proposal=proposal,
                    num_simulations=num_simulations,
                    num_workers=num_workers,
                )
                
                print('x', x)

                neural_dens = inf.append_simulations(theta, x, proposal).train()


                posterior = inf.build_posterior(neural_dens)



                proposal = posterior.set_default_x(obs_real)

    


            posterior_snpe = posterior

            posterior_snpe_list.append(posterior_snpe)
            
        list_collection.append(posterior_snpe_list)


    # In[ ]:
    end_time = datetime.datetime.now()


    diff_time = end_time - start_time

    import json

    json_dict = {
    "CPU time for step:": str(diff_time)}
    with open( "time_snpe_nsf.json", "a") as f:
        json.dump(json_dict, f)
        f.close()

    torch.save(list_collection, 'list_collection.pt')


    # In[8]:


    list_collection = torch.load('list_collection.pt')


    # In[9]:


    list_collection


    # ### For incremental approach: 

    # In[ ]:


    range_list = [5,10, 15]



    list_collection_inc = []

    start_time = datetime.datetime.now()

    for i in range(10):
        
        np.random.seed(i)

        posterior_incremental_list = []


        for num_simulations in num_simulations_list:

            prior_i = utils.torchutils.BoxUniform(low=prior_min[0:range_list[0]], high = prior_max[0:range_list[0]])

            
            simulator_stats, prior_i = prepare_for_sbi(Gaussian, prior_i)
            
            inf = SNPE(prior_i, density_estimator=density_estimator)
            
            proposal = prior_i

            start_num = 1

            for index in range(len(range_list)-1):

                ## i defines number of parameters to be inferred, j indicates how many parameters 
                #to come in the next round
                i = range_list[index]
                j = range_list[index+1]

                print(i, j)

                num_sim = int(num_simulations * (start_num / 10))

                start_num += 9

                start_time = datetime.datetime.now()

                theta, x =  simulate_for_sbi(
                    simulator_stats,
                    proposal=proposal,
                    num_simulations=num_simulations,
                    num_workers=num_workers,

                )

                inf = inf.append_simulations(theta, x)
                neural_dens = inf.train()

                posterior = inf.build_posterior(neural_dens)

                if i < 2:
                    obs_real = Gaussian([true_thetas[0, 0:i]])

                else:
                    obs_real = Gaussian(true_thetas[0, 0:i])


                proposal1 = posterior.set_default_x(obs_real)

                next_prior = utils.torchutils.BoxUniform(low=prior_min[i:j], high=prior_max[i:j])

                combined_prior = Combined(proposal1, next_prior, number_params_1=i)


                ## set inf for next round:
                inf = SNPE(combined_prior, density_estimator=density_estimator)


                ## set combined prior to be the new prior_i:
                proposal= combined_prior

                finish_time = datetime.datetime.now()

                diff = finish_time - start_time

                print('took ', diff, ' for this step')

            num_sim = int(num_simulations * (start_num / 10))

            theta, x =  simulate_for_sbi(
                simulator_stats,
                proposal=proposal,
                num_simulations=num_sim,
                num_workers=num_workers,

            )

            inf = inf.append_simulations(theta, x)
            neural_dens = inf.train()

            posterior_incremental = inf.build_posterior(neural_dens) 

            posterior_incremental_list.append(posterior_incremental)
            
        list_collection_inc.append(posterior_incremental_list)


    # In[50]:

    end_time = datetime.datetime.now()


    diff_time = end_time - start_time


    json_dict = {
    "CPU time for step:": str(diff_time)}
    with open( "time_incremental_nsf.json", "a") as f:
        json.dump(json_dict, f)
        f.close()


    torch.save(list_collection_inc, 'list_collection_inc.pt')


    # In[46]:


    list_collection_inc = torch.load('list_collection_inc.pt')


    # In[ ]:


    import torch.nn.functional as F


    #out = F.kl_div(analytic_sample, posterior_sample)



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




 


    import torch

    analytic = torch.distributions.normal.Normal(true_thetas, 1)

    analytic.stddev


    # In[12]:





    analytic = torch.distributions.normal.Normal(true_thetas, 1)


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



    analytic = torch.distributions.normal.Normal(true_thetas, 1)


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


    lower_incremental = mean_incremental - [element * 1.00 for element in stdev_incremental]

    upper_incremental = mean_incremental + [element * 1.00 for element in stdev_incremental]


    # In[66]:


    mean_snpe = np.mean(np.array(overall_snpe_list), axis=0)

    print(mean_snpe)

    stdev_snpe = np.std(np.array(overall_snpe_list), axis=0)

    print(stdev_snpe)


    lower_snpe = mean_snpe - [element * 1.00 for element in stdev_snpe]

    upper_snpe = mean_snpe + [element * 1.00 for element in stdev_snpe]


    # ### Compare KL-divergence of snpe approach with incremental approach in a plot:
    # 
    # #### x = number of simulations per round/step

    # In[67]:


    figure_mosaic = """
    ACC
    BCC
    """

    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(11, 8))

        

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

    plt.savefig('Gauss_plot_1stddev_nsf_noprop.png')


    #axes['B'].set_xticklabels(['0k','2k', '4k', '6k', '8k', '10k'])
    #axes['A'].set_xticklabels(['0k','2k', '4k', '6k', '8k', '10k'])
    #axes['C'].set_xticklabels(['0k','2k', '4k', '6k', '8k', '10k'])
    #plt.xticks(['1k', '3k', '5k', '10k'])


    mean_incremental = np.log(mean_incremental)


    stdev_incremental = np.log(stdev_incremental)



    lower_incremental = np.log(lower_incremental)

    upper_incremental = np.log(upper_incremental)


    mean_snpe = np.log(mean_snpe)



    stdev_snpe = np.log(stdev_snpe)




    lower_snpe = np.log(lower_snpe)

    upper_snpe = np.log(upper_snpe)




    figure_mosaic = """
    ACC
    BCC
    """

    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(11, 8))

        

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

    plt.savefig('Gauss_plot_1stddev_log_nsf_noprob.png')

if __name__ == "__main__":
    torch.manual_seed(5)
    np.random.seed(5)
    main(sys.argv[1:])