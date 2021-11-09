#!/usr/bin/env python
# coding: utf-8

import os.path as op
import tempfile

import matplotlib.pyplot as plt


import hnn_core
from hnn_core import simulate_dipole, jones_2009_model
from hnn_core.viz import plot_dipole



from summary_features.calculate_summary_features import calculate_summary_stats, calculate_summary_statistics_alternative

import numpy as np
import torch

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi



from utils.simulation_wrapper import event_seed, set_network_default


import matplotlib.gridspec as gridspec

from utils import inference
from utils.simulation_wrapper import event_seed, simulation_wrapper
from utils.helpers import get_time

from data_load_writer import write_to_file


import sys

def main(argv):
    """
    description: simulated and inferes with sbi; plotting the histogram of the summary statistics for both prior and posterior

    arguments:
    arg 1: number of simulations
    arg 2: type of density estimator; default is nsf
    arg 3: number of cpus that should be used
    arg 4: number of samples that should be drawn from posterior
    
    """

    try:
        number_simulations = int(argv[0])
    except:
        number_simulations = 500
    try:
        density_estimator = argv[1]
    except:
        density_estimator = 'nsf'
    try:
        num_workers = int(argv[2])
    except:
        num_workers = 8

    try:
        num_samples = int(argv[3])
    except:
        num_samples = 100


    start_time = get_time()

    true_params = torch.tensor([63.53, 137.12])
    file_writer = write_to_file.WriteToFile(experiment='ERP_{}_num_params:{}_'.format(density_estimator, true_params.size(dim=0)), num_sim=number_simulations,
                    true_params=true_params, density_estimator=density_estimator)


    prior_min = [43.8, 89.49]   # 't_evdist_1', 'sigma_t_evdist_1', 't_evprox_2', 'sigma_t_evprox_2'

    prior_max = [79.9, 152.96]  

    prior = utils.torchutils.BoxUniform(low=prior_min, 
                                        high=prior_max)


    s_real = inference.run_only_sim(true_params)

    print(s_real)

    posterior, theta, x = inference.run_sim_inference(prior, simulation_wrapper, number_simulations, density_estimator=density_estimator, num_workers=num_workers)







    fig = plt.figure(figsize=(2,40), frameon = False, dpi = 100, tight_layout=True)

    gs = gridspec.GridSpec(nrows=18, ncols=1)


    for i in range(x.size(dim=1)):
        print(i)
        
        globals()['ax%s' % i] = fig.add_subplot(gs[i])

        globals()['sum_stats%s' % i] = []

        for j in range(x.size(dim=0)):
            globals()['sum_stats%s' % i].append(x[j][i])



        globals()['ax%s' % i].hist(globals()['sum_stats%s' % i], bins=20, density=True, facecolor='g', alpha=0.75, histtype='barstacked')
        globals()['ax%s' % i].set_title('Summary stat {} (from simulation)'.format(i))
        globals()['ax%s' % i].axvline(s_real[i], color='red', label='true obs')
        globals()['ax%s' % i].legend(loc='upper right')




    file_writer.save_fig('Histograms_from_prior.pdf')



    samples = posterior.sample((num_samples,), 
                            x=s_real)





    s_x = inference.run_only_sim(samples)




    fig = plt.figure(figsize=(20,5))

    gs = gridspec.GridSpec(nrows=1, ncols=2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    for sample in s_x:
        
        #assert torch.equal(sample, s_real[0])
        ax0.plot(sample.detach().numpy()[:10])
        ax0.set_title('Summary statistics 1-10 of samples')
        ax0.set(ylim=(-500, 7000))
    

    ax1.plot(s_real[0][:10])
    ax1.set_title('Summary statistics 1-10 of real parameters')
    ax1.set(ylim=(-500, 7000))

    plt.savefig('summary_stats1')


    fig = plt.figure(figsize=(20,5))

    gs = gridspec.GridSpec(nrows=1, ncols=2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    for sample in s_x:
        
        #assert torch.equal(sample, s_real[0])
        ax0.plot(sample.detach().numpy()[10:])
        ax0.set_title('Summary statistics 10-18 of samples')
        #ax0.set(ylim=(-500, 7000))
    

    ax1.plot(s_real[0][10:])
    ax1.set_title('Summary statistics 10-18 of real parameters')
    #ax1.set(ylim=(-500, 7000))
    
    file_writer.save_fig('summary_stats2')


    # In[50]:


    import math
    import numpy as np

    fig = plt.figure(figsize=(20,400))

    gs = gridspec.GridSpec(nrows=len(s_x)-1, ncols=1)




    for i in range(len(s_x[0])-1):
        
        globals()['ax%s' % i] = fig.add_subplot(gs[i])

        globals()['sum_stats%s' % i] = []

        for j in range(len(s_x)-1):
            globals()['sum_stats%s' % i].append(s_x[j][i])



        globals()['ax%s' % i].hist(globals()['sum_stats%s' % i], bins=20, density=True, facecolor='g', alpha=0.75, histtype='barstacked')
        globals()['ax%s' % i].set_title('Histogram of summary stat {} (drawn 100 samples)'.format(i))
        #ax0.set(ylim=(-500, 7000))

        globals()['ax%s' % i].axvline(s_real[i], color='red', label='true obs')
        globals()['ax%s' % i].legend(loc='upper right')




    plt.savefig('Histograms_sumstats_from_posterior.pdf')



    finish_time = get_time()
    file_writer.save_all(posterior, prior, theta=theta, x =x, fig=fig, start_time=start_time, finish_time=finish_time)





if __name__ == "__main__":
   main(sys.argv[1:])
