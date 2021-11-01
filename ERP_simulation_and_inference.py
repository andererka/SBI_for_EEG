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


## defining neuronal network model

net = jones_2009_model()
#net.plot_cells()
#net.cell_types['L5_pyramidal'].plot_morphology()

## defining weights

weights_ampa_d1 = {'L2_basket': 0.006562, 'L2_pyramidal': .000007,
                   'L5_pyramidal': 0.142300}
weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317,
                   'L5_pyramidal': 0.080074}
synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
net.add_evoked_drive(
    'evdist1', mu=63.53, sigma=3.85, numspikes=1, weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1, location='distal',
    synaptic_delays=synaptic_delays_d1, event_seed=4)




weights_ampa_p1 = {'L2_basket': 0.08831, 'L2_pyramidal': 0.01525,
                   'L5_basket': 0.19934, 'L5_pyramidal': 0.00865}
synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                        'L5_basket': 1., 'L5_pyramidal': 1.}

# all NMDA weights are zero; pass None explicitly
net.add_evoked_drive(
    'evprox1', mu=26.61, sigma=2.47, numspikes=1, weights_ampa=weights_ampa_p1,
    weights_nmda=None, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=4)

# Second proximal evoked drive. NB: only AMPA weights differ from first
weights_ampa_p2 = {'L2_basket': 0.000003, 'L2_pyramidal': 1.438840,
                   'L5_basket': 0.008958, 'L5_pyramidal': 0.684013}
# all NMDA weights are zero; omit weights_nmda (defaults to None)
net.add_evoked_drive(
    'evprox2', mu=137.12, sigma=8.33, numspikes=1,
    weights_ampa=weights_ampa_p2, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=4)


def simulation_wrapper(params):   #input possibly array of 1 or more params
    """
    Returns summary statistics from conductance values in `params`.

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    
    
    window_len, scaling_factor = 30, 3000
    net._params['t_evdist_1'] = params[0]
    net._params['sigma_t_evdist_1'] = params[1]
    net._params['t_evprox_2'] = params[2]
    net._params['sigma_t_evprox_2'] = params[3]

    ##simulates 8 trials at a time like this
    dpls = simulate_dipole(net, tstop=170., n_trials=1)
    for dpl in dpls:
        obs = dpl.smooth(window_len).scale(scaling_factor).data['agg']

    #left out summary statistics for a start
    sum_stats = calculate_summary_stats(torch.from_numpy(obs))
    return sum_stats




window_len = 30
prior_min = [1.0, 1 , 1.0, 3]   # 't_evdist_1', 'sigma_t_evdist_1', 't_evprox_2', 'sigma_t_evprox_2'

prior_max = [175.0, 8, 175, 15]  

prior = utils.torchutils.BoxUniform(low=prior_min, 
                                    high=prior_max)

number_simulations = 500



def run_simulation_inference(prior, simulation_wrapper, num_simulations=1000):


    #posterior = infer(simulation_wrapper, prior, method='SNPE_C', 
                  #num_simulations=number_simulations, num_workers=4)     

    simulator, prior = prepare_for_sbi(simulation_wrapper, prior)
    inference = SNPE(prior)


    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_simulations)

    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator) 

    return posterior, theta, x

posterior, theta, x = run_simulation_inference(prior, simulation_wrapper, number_simulations)

window_len, scaling_factor = 30, 3000


## set params as the 'true parameters'
net._params['t_evdist_1'] = 63.53
net._params['sigma_t_evdist_1'] = 3.85
net._params['t_evdist_2'] = 137.12
net._params['sigma_t_evprox_2'] = 8.33

dpls = simulate_dipole(net, tstop=170., n_trials=1)
for dpl in dpls:
    obs = dpl.smooth(window_len).scale(scaling_factor).data['agg']

obs_real = calculate_summary_stats(torch.from_numpy(obs))

samples = posterior.sample((1000,), 
                           x=obs_real)




true_params = torch.Tensor([63.53, 3.85, 137.12, 8.33])




fig, axes = analysis.pairplot(samples,
                           #limits=[[.5,80], [1e-4,15.]],
                           #ticks=[[.5,80], [1e-4,15.]],
                           figsize=(5,5),
                           points=true_params,
                           points_offdiag={'markersize': 6},
                           points_colors='r');


from dataloads_storage import write_to_file

file_writer = write_to_file.WriteToFile(experiment='ERP_results', num_sim=number_simulations,
                true_params=true_params)

file_writer.save_posterior(posterior)

file_writer.save_prior(prior)

file_writer.save_thetas(theta)

file_writer.save_observations(x)

file_writer.save_fig(fig)

file_writer.save_class()
