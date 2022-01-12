from data_load_writer import load_from_file as lf
from data_load_writer import write_to_file
import matplotlib.pyplot as plt


import hnn_core
from hnn_core import simulate_dipole, jones_2009_model
from hnn_core.viz import plot_dipole

# import the summary statistics that you want to investigate
from summary_features.calculate_summary_features import calculate_summary_statistics_alternative as extract_sumstats
#from summary_features.calculate_summary_features import calculate_summary_stats_temporal as extract_sumstats
from summary_features.calculate_summary_features import calculate_summary_stats_number as number_sumstats


import numpy as np
import torch

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.plot import cov, compare_vars, plot_varchanges

from utils.plot import plot_KLs

# sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE_C, prepare_for_sbi, simulate_for_sbi

from utils import inference

from utils.helpers import SummaryNet
import sys

from data_load_writer import write_to_file


import os
import pickle

import sys

sys.path.append('/mnt/qb/work/macke/kanderer29')

from utils.helpers import get_time


## defining neuronal network model

from utils.simulation_wrapper import event_seed, set_network_default, simulation_wrapper,simulation_wrapper_obs

from joblib import Parallel, delayed

import matplotlib.gridspec as gridspec



"""
This file aims to investigate different summary statistics, their time efficiency and how well they represent the data.
"""



def main(argv):

    start = get_time()

    slurm = True

    try:
        experiment_name = argv[0]
    except:
        experiment_name = '6_sum_stats_500sim_3params'

    ### loading the class:
    #with open('results/{}/class'.format(file), "rb") as pickle_file:
        #file_writer = pickle.load(pickle_file)


    try:
        num_workers = int(argv[1])
    except:
        num_workers = 8
    try:
        num_sim = int(argv[2])
    except:
        num_sim = 100

    try:
        num_samples = int(argv[3])

    except:
        num_samples = 100

    true_params = torch.tensor([[26.61, 63.53,  137.12]])
    sim_wrapper = simulation_wrapper_obs

    print('okay')

    os.chdir('')

    os.chdir('/mnt/qb/work/macke/kanderer29')

    print('slurm1', slurm)

    print(os.getcwd())



    print('new experiment')
    file_writer = write_to_file.WriteToFile(
    experiment=experiment_name,
    num_sim=num_sim,
    true_params=true_params,
    density_estimator='nsf',
    num_params=3,
    num_samples=num_samples,
    )

    try:
        os.mkdir(file_writer.folder)
    except:
        print('file exists')

    

    prior_min = [7.9, 43.8,  89.49] 

    prior_max = [30, 79.9, 152.96]

    prior = utils.torchutils.BoxUniform(low=prior_min, high=prior_max)

    inf = SNPE_C(prior, density_estimator="nsf")

    try:
                
        print('line 141', os.getcwd())
        theta = torch.load('results/{}/thetas.pt'.format(experiment_name))
        x_without = torch.load('results/{}/obs_without.pt'.format(experiment_name))

    except:
        print('running')
        theta, x_without = inference.run_sim_theta_x(
            prior,
            simulation_wrapper=sim_wrapper,
            num_simulations=num_sim,
            num_workers=num_workers
        )
       

        
    ## save thetas and x_without to file_writer:
    file_writer.save_obs_without(x_without)
    file_writer.save_thetas(theta)

    print('line159')

    torch.save(file_writer, "{}/class.pt".format(file_writer.folder))
    
    ## 21 summary features:
    x_21 = number_sumstats(x_without, 21)
    inf = SNPE_C(prior=prior, density_estimator = 'nsf')
    inf = inf.append_simulations(theta, x_21)
    density_estimator = inf.train()
    posterior_21 = inf.build_posterior(density_estimator)

    ## 17 summary features:

    x_17 = number_sumstats(x_without, 17)
    inf = SNPE_C(prior=prior, density_estimator = 'nsf')
    inf = inf.append_simulations(theta, x_17.to(torch.float32))
    density_estimator = inf.train()
    posterior_17 = inf.build_posterior(density_estimator)

    print('line 181')

    obs_real = inference.run_only_sim(true_params, sim_wrapper)

    obs_real_number21 = number_sumstats(obs_real, 21)
    obs_real_number17 = number_sumstats(obs_real, 17)


    samples_number17 = posterior_17.sample((num_samples,), x=obs_real_number17)

    print('line 191')

    samples_number21 = posterior_21.sample((num_samples,), x=obs_real_number21)

    ### sample from prior now
    samples_prior = []


    for i in range(num_samples):
        sample = prior.sample()
        samples_prior.append(sample)

    ## simulate from posterior samples:
    s_x_17 = inference.run_only_sim(samples_number17, sim_wrapper, num_workers)
    s_x_21 = inference.run_only_sim(samples_number21, sim_wrapper, num_workers)

    t = obs_real_number21[0]


    sample_batch_21 = []
    batch_size = 10

    print('line 213')

    for i in range(batch_size):

        sample21_list = []

        for i in range(21):
            x = number_sumstats(x_without, 21)
            x_c = torch.cat((x[:,:i], x[:,i+1:]), axis = 1)
            inf = SNPE_C(prior=prior, density_estimator = 'nsf')
            inf = inf.append_simulations(theta, x_c)
            density_estimator = inf.train()
            posterior = inf.build_posterior(density_estimator)
            globals()['samples21_%s' % i] = posterior.sample((num_samples,), x=torch.cat((t[:i], t[i+1:]), axis = 0))
            
            sample21_list.append(globals()['samples21_%s' % i] )
        
        sample_batch_21.append(sample21_list)

    sum_stats_names = ['max', 'min', 'peak_to_peak', 'area', 'autocorr', 'zero_cross',
                    'max1', 'min1', 'peak_to_peak1', 'area1', 'autocorr1', 
                                            'max2', 'min2', 'peak_to_peak2', 'area2', 'autocorr2', 
                                            'max3', 'min3', 'peak_to_peak3', 'area3', 'autocorr3']


    print('line 238')

    im = plot_varchanges(sample_batch_21, samples_number21, xticklabels=sum_stats_names, yticklabels=["t_evprox_1", "t_evdist_1", "t_evprox_2"], plot_label='', batchsize=0)

    file_writer.save_fig(im, 'var_changes21')


    t = obs_real_number17[0]
    sample_batch_17 = []


    for i in range(batch_size):

        sample17_list = []

        for i in range(21):
            x = number_sumstats(x_without, 17)
            x_c = torch.cat((x[:,:i], x[:,i+1:]), axis = 1)
            inf = SNPE_C(prior=prior, density_estimator = 'nsf')
            inf = inf.append_simulations(theta, x_c)
            density_estimator = inf.train()
            posterior = inf.build_posterior(density_estimator)
            globals()['samples21_%s' % i] = posterior.sample((num_samples,), x=torch.cat((t[:i], t[i+1:]), axis = 0))
            
            sample17_list.append(globals()['samples21_%s' % i] )
        
        sample_batch_17.append(sample21_list)

    sum_stats_names17 =  [
                    'arg_p50',
                    'arg_N100',
                    'arg_P200',
                    'p50',
                    'N100',
                    'P200',
                    'p50_moment1',
                    'N100_moment1',
                    'P200_moment1',
                    'p50_moment2',
                    'N100_moment1',
                    'P200_moment1',
                    'area_p50',
                    'area_N100',
                    'area_P200',
                    'mean4000',
                    'mean1000'
                ]


    im = plot_varchanges(sample_batch_17, samples_number17, xticklabels=sum_stats_names17, yticklabels=["t_evprox_1", "t_evdist_1", "t_evprox_2"], plot_label='', batchsize=0)

    file_writer.save_fig(im, 'var_changes17')



    fig, axes = plt.subplots(1, 1, figsize=(30, 10), sharex=True)

    print('line 295')

    plot_KLs(sample_batch_17,
            samples_number17,
            idx=0,
            batchsize=10,
            kind='box',
            agg_with='mean'
        )


    axes.set_xlabel("missing feature", size=14)
    axes.set_xticklabels(sum_stats_names17)
    axes.tick_params(axis="both", which="major", labelsize=12)
    ylabel = axes.get_ylabel()
    axes.set_ylabel(ylabel, size=14)

    file_writer.save_fig(fig, 'KL_17_features')

    fig, axes = plt.subplots(1, 1, figsize=(30, 10), sharex=True)


    plot_KLs(sample_batch_21,
            samples_number21,
            idx=0,
            batchsize=10,
            kind='box',
            agg_with='mean'
        )


    axes.set_xlabel("missing feature", size=14)
    axes.set_xticklabels(sum_stats_names)
    axes.tick_params(axis="both", which="major", labelsize=12)
    ylabel = axes.get_ylabel()
    axes.set_ylabel(ylabel, size=14)

    file_writer.save_fig(fig, 'KL_21_features')

    s_x_prior = inference.run_only_sim(samples_prior, sim_wrapper, num_workers)

    fig1, ax = plt.subplots(1, 1)
    ax.set_title("Simulating from posterior")
    for s in s_x_21:
        plt.plot(s, alpha=0.2, color='blue')
        #plt.ylim(-30,30)
        plt.xlim(0, 7000)
        
    plt.plot(obs_real_number21[0], label='Ground truth', color='red')

    file_writer.save_fig(fig1, 'from_posterior_21features')
        
    fig2, ax = plt.subplots(1, 1)
        
    for s in s_x_17:
        plt.plot(s, alpha=0.2, color='blue')
        #plt.ylim(-30,30)
        plt.xlim(0, 7000)
        
    plt.plot(obs_real_number17[0], label='Ground truth', color='red')

    file_writer.save_fig(fig2, 'from_posterior_17features')

    fig3, ax = plt.subplots(1, 1)
    ax.set_title("Simulating from prior")
    for x_w in s_x_prior:
        plt.plot(x_w, alpha=0.2, color='blue')

    file_writer.save_fig(fig3, 'from_prior')

if __name__ == "__main__":
    main(sys.argv[1:])
