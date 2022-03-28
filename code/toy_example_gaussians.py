from random import gauss
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

import datetime

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

from sbi.utils import RestrictionEstimator

from utils import inference


import pickle
import sys

import os

## gaussian function for simulator
def Gaussian(thetas, normal_noise=1):
    
    np.random.seed(np.random.choice(1000))
    
    gauss_list = []
    
    for i, theta in enumerate(thetas):
    
        mu, sigma = theta, normal_noise # mean and standard deviation

        s = np.random.normal(mu, sigma, 1)
    
        
        gauss_list.append(s[0])

        
    gauss_obs = torch.tensor(gauss_list)
    
    return gauss_obs


## main function to compare nulti-round SNPE with NIPE approach
def main(argv):
    '''
    takes the following arguments:

    - density_estimator: can be 'maf', 'nsf' or 'mdn' (or other). Default is 'mdn'
    - num_workers: number of cpus available. Default is 8
    - slurm: takes 0 or 1. 0 if not on slurm. If 1, results get stored under specific result folder (specified in 'write_to_file.WriteToFile')
    - experiment_name: defines name of result folder. If not defined, then it's 'toy_example_mdn'
    - rounds: in order to get a valid comparison, multiple posteriors are calculated in multiple rounds. Default is 10. Argument defines how 
            many posteriors are calculated for each approach. Later, KL divergence measures are calculated and mean and stddev of the KL div between 
            inferred and analytic posteriors are plotted
    - set_proposal: takes 0 or 1. If 1, proposal is set in 'append_simulations' for the multi-round approach. This is the 'proper' approach
                    because then leakage correction can be applied. Without it tough, inference can be faster. 
                    It's not working for mdn's (there is probably an issue with numerical instability -see https://github.com/mackelab/sbi/issues/669#issue-1177918978)
    - ratio: depending on if the already inferred posteriors are taken ONLY for simulation, but thetas of these subsets are not inferred 
                again in the nect round, it makes sense to use a smaller number of simulations in the first round and a larger number for the
                last - then the quality of inference is not so different between subsets.
                If inferred posteriors are only taken for simulation, this argument should be set to 0 (false).

    '''


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
        experiment_name = "toy_example_mdn"
    try:
        rounds = int(argv[4])
    except:
        rounds = 10
    try:
        set_proposal = bool(int(argv[5]))
    except:
        set_proposal = False

    ## Ratio only makes sense if we do not make seperate inference on the subsets, but instead make inference again on the 
    ## inferred posteriors such that it gets more and more restricted.
    
    try:
        ratio = bool(int(argv[2]))
    except:
        ratio = False


    file_writer = write_to_file.WriteToFile(
    experiment=experiment_name,
    density_estimator=density_estimator,
    slurm=slurm,
    )

    try:
        os.mkdir(file_writer.folder)
    except:
        print('file exists')

    # stores the running file into the result folder for later reference:
    open('{}/toy_example_gaussians.py'.format(file_writer.folder), 'a').close()
    shutil.copyfile(str(os.getcwd() + '/toy_example_gaussians.py'), str(file_writer.folder+ '/toy_example_gaussians.py'))



    os.chdir(file_writer.folder)

    json_dict = {
    "arguments:": str(argv)}
    with open( "argument_list.json", "a") as f:
        json.dump(json_dict, f)
        f.close()

        


    # ### Larger comparison with KL-divergence between analytic and inferred posterior

    # ### Calculate posterior for different number of simulations: 1k,  3k, 5k, 10k

    # ### starting with multi-round snpe


    import torch


    true_thetas = torch.tensor([[3.0, 6.0, 20.0, 10.0, 90.0, 55.0, 27.0, 27.0, 4.0, 70.0, 5.0, 66.0, 99.0, 40.0, 45.0]])
    parameter_names = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15']

    prior_max = [100.0] * 15
    prior_min = [1.0] * 15


    # In[6]:


    num_simulations_list = [200, 500, 750, 1000, 1500, 2000]

    
    #num_simulations_list = [200]




    obs_real = Gaussian(true_thetas[0])


    torch.manual_seed(4)


    list_collection = []


    start = datetime.datetime.now()

    for _ in range(rounds):
        

        posterior_snpe_list = []

        for num_simulations in num_simulations_list:
            
            prior = utils.torchutils.BoxUniform(low=prior_min, high = prior_max)
            simulator_stats, prior = prepare_for_sbi(Gaussian, prior)
            
            inf = SNPE(prior, density_estimator=density_estimator)
            
            proposal = prior

            posteriors = []

            for j in range(3):

                print('round: ', j)
                
                theta, x = simulate_for_sbi(
                    simulator_stats,
                    proposal=proposal,
                    num_simulations=num_simulations,
                    num_workers=num_workers,
                )
                
                if set_proposal:
                    neural_dens = inf.append_simulations(theta, x, proposal).train()
                else:
                    neural_dens = inf.append_simulations(theta, x).train()


                posterior = inf.build_posterior(neural_dens)

                posteriors.append(posterior)



                proposal = posterior.set_default_x(obs_real)

            torch.save(posteriors, 'posteriors_each_round.pt')


            posterior_snpe = posterior

            posterior_snpe_list.append(posterior_snpe)
            
        list_collection.append(posterior_snpe_list)

        
    end = datetime.datetime.now()

    diff  = end - start

    print(diff)


    json_dict = {
    "CPU time for step:": str(diff)}
    with open( "time_snpe_nsf.json", "a") as f:
        json.dump(json_dict, f)
        f.close()

    torch.save(list_collection, 'list_collection.pt')


    # In[8]:


    list_collection = torch.load('list_collection.pt')



    range_list = [5, 10, 15]



    list_collection_inc = []

    start_time = datetime.datetime.now()

    for _ in range(rounds):
        

        posterior_incremental_list = []


        for num_simulations in num_simulations_list:

            prior_i = utils.torchutils.BoxUniform(low=prior_min[0:range_list[0]], high = prior_max[0:range_list[0]])

            
            simulator_stats, prior_i = prepare_for_sbi(Gaussian, prior_i)
            
            inf = SNPE(prior_i, density_estimator=density_estimator)
            
            proposal = prior_i

            start_num = 1

            previous_i = 0
            proposal_list = []

            for index in range(len(range_list)-1):

                ## i defines number of parameters to be inferred, j indicates how many parameters 
                #to come in the next round
                i = range_list[index]
                j = range_list[index+1]

                print(i, j)

                if ratio == True:

                    num_sim = int(num_simulations * (start_num / 10))

                    start_num += 9
                else:
                    num_sim = num_simulations


                theta, x =  simulate_for_sbi(
                    simulator_stats,
                    proposal=proposal,
                    num_simulations=num_sim,
                    num_workers=num_workers,

                )



                theta = theta[:, previous_i:i]
                x = x[:, previous_i:i]



                
                inf = inf.append_simulations(theta, x)
                neural_dens = inf.train()

                posterior = inf.build_posterior(neural_dens)

                obs_real2 = obs_real[previous_i:i]



                proposal1 = posterior.set_default_x(obs_real2)



                next_prior = utils.torchutils.BoxUniform(low=prior_min[i:j], high=prior_max[i:j])

                proposal_list.append(proposal1)

                combined_prior = Combined(proposal_list, next_prior, steps=[0,5,10,15])


                ## here we only make inference on the next prior, not the whole set so far
                inf = SNPE(next_prior, density_estimator=density_estimator)


                ## set combined prior to be the new proposal:
                proposal= combined_prior

                previous_i = i



            if ratio == True:

                num_sim = int(num_simulations * (start_num / 10))

            else:
                num_sim = num_simulations



            theta, x =  simulate_for_sbi(
                simulator_stats,
                proposal=proposal,
                num_simulations=num_sim,
                num_workers=num_workers,

            )



            theta = theta[:,previous_i:]
            x = x[:,previous_i:]



            inf = inf.append_simulations(theta, x)
            neural_dens = inf.train()

            posterior_incremental = inf.build_posterior(neural_dens)

            obs_real2 = obs_real[previous_i:]


            posterior_incremental.set_default_x(obs_real2)


            proposal_list.append(posterior_incremental)

            combined_posterior = Combined(proposal_list, None, steps=[0,5,10,15])

            posterior_incremental_list.append(combined_posterior)
            
        list_collection_inc.append(posterior_incremental_list)



    end_time = datetime.datetime.now()


    diff_time = end_time - start_time


    json_dict = {
    "CPU time for step:": str(diff_time)}
    with open( "time_incremental_nsf.json", "a") as f:
        json.dump(json_dict, f)
        f.close()


    torch.save(list_collection_inc, 'list_collection_inc.pt')




    list_collection_inc = torch.load('list_collection_inc.pt')


    import torch.nn.functional as F



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


        


    analytic = torch.distributions.normal.Normal(true_thetas, 1)


    overall_incremental_list = []

    for posterior_incremental_list in list_collection_inc:
        
        KL_incremental = []

        for posterior_incremental in posterior_incremental_list:



            KL = KL_Gauss(posterior_incremental, analytic)


            KL_incremental.append(KL)

            
        overall_incremental_list.append(KL_incremental)

        



    mean_incremental = np.mean(np.array(overall_incremental_list), axis=0)

    stdev_incremental = np.std(np.array(overall_incremental_list), axis=0)

    lower_incremental = mean_incremental - [element * 0.5 for element in stdev_incremental]

    upper_incremental = mean_incremental + [element * 0.5 for element in stdev_incremental]


    # In[66]:


    mean_snpe = np.mean(np.array(overall_snpe_list), axis=0)

    stdev_snpe = np.std(np.array(overall_snpe_list), axis=0)

    lower_snpe = mean_snpe - [element * 0.5 for element in stdev_snpe]

    upper_snpe = mean_snpe + [element * 0.5 for element in stdev_snpe]


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

    plt.savefig('Gauss_plot_1stddev.png')


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

    plt.savefig('Gauss_plot_1stddev_log.png')

if __name__ == "__main__":
    torch.manual_seed(5)
    np.random.seed(5)
    main(sys.argv[1:])
