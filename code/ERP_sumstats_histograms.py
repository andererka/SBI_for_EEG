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
from summary_features.calculate_summary_features import calculate_summary_stats


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

    true_params = torch.tensor([[63.53, 137.12]])

    #writes to result folder
    file_writer = write_to_file.WriteToFile(experiment='ERP_{}'.format(density_estimator), num_sim=number_simulations,
                    true_params=true_params, density_estimator=density_estimator)


    prior_min = [43.8, 89.49]   # 't_evdist_1', 'sigma_t_evdist_1', 't_evprox_2', 'sigma_t_evprox_2'

    prior_max = [79.9, 152.96]  

    prior = utils.torchutils.BoxUniform(low=prior_min, 
                                        high=prior_max)


    s_real = inference.run_only_sim(true_params)[0]
    


    posterior, theta, x = inference.run_sim_inference(prior, simulation_wrapper, number_simulations, density_estimator=density_estimator, num_workers=num_workers)

    sum_stats_names = ['value_p50', 'value_N100', 'value_P200', 'value_arg_p50', 'arg_N100', 'arg_P200',
    'p50_moment1', 'p50_moment2', 'p50_moment3', 'p50_moment4',
    'N100_moment1', 'N100_moment2', 'N100_moment3', 'N100_moment4',
    'P200_moment1','P200_moment2', 'P200_moment3', 'P200_moment4'
            ]

    finish_time = get_time()



    ### creating histogram that shows the summary statistics predictions of the observations ###
    fig = plt.figure(figsize=(2,40), frameon = False, dpi = 100, tight_layout=True)

    gs = gridspec.GridSpec(nrows=18, ncols=1)


    for i in range(x.size(dim=1)):
        print(i)
        
        globals()['ax%s' % i] = fig.add_subplot(gs[i])

        globals()['sum_stats%s' % i] = []

        for j in range(x.size(dim=0)):
            globals()['sum_stats%s' % i].append(x[j][i])



        globals()['ax%s' % i].hist(globals()['sum_stats%s' % i], bins=20, density=True, facecolor='g', alpha=0.75, histtype='barstacked')
        globals()['ax%s' % i].set_title('Summary stat {} (from simulation)'.format(sum_stats_names[i]))
        globals()['ax%s' % i].axvline(s_real[i], color='red', label='true obs')
        globals()['ax%s' % i].legend(loc='upper right')


    fig.suptitle('Summary stats histogram from simulated obs.', fontsize=16)

    file_writer.save_fig(fig)


    ### samples from posterior to then with 'run_only_sim' make a prediction how our observation would look like
    ### given the sampled parameter values.
    ### this is visualized in histogram plots for every single summary statistics

    samples = posterior.sample((num_samples,), 
                            x=s_real)
    print('here', samples)


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
    

    ax1.plot(s_real[:10])
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
    

    ax1.plot(s_real[10:])
    ax1.set_title('Summary statistics 10-18 of real parameters')
    #ax1.set(ylim=(-500, 7000))
    
    file_writer.save_fig('summary_stats2')


    # In[50]:


    import math
    import numpy as np

    fig = plt.figure(figsize=(2,40))

    gs = gridspec.GridSpec(nrows=len(s_x)-1, ncols=1)




    for i in range(len(s_x[0])-1):
        
        globals()['ax%s' % i] = fig.add_subplot(gs[i])

        globals()['sum_stats%s' % i] = []

        for j in range(len(s_x)-1):
            globals()['sum_stats%s' % i].append(s_x[j][i])



        globals()['ax%s' % i].hist(globals()['sum_stats%s' % i], bins=20, density=True, facecolor='g', alpha=0.75)
        globals()['ax%s' % i].set_title('Histogram of summary stat {} (drawn 100 samples)'.format(i))
        #ax0.set(ylim=(-500, 7000))

        globals()['ax%s' % i].axvline(s_real[i], color='red', label='true obs')
        globals()['ax%s' % i].legend(loc='upper right')


    fig.suptitle('Summary stats histogram from posterior predictions.', fontsize=16)

    file_writer.save_fig(fig)


    file_writer.save_all(posterior, prior, theta=theta, x =x, start_time=start_time, finish_time=finish_time)





if __name__ == "__main__":
   main(sys.argv[1:])
