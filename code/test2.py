#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.plot import conditional_pairplot_comparison

# sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi



# import the summary statistics that you want to investigate
from summary_features.calculate_summary_features import calculate_summary_statistics_alternative as alternative_sumstats
from summary_features.calculate_summary_features import calculate_summary_stats_temporal as temporal_sumstats
from summary_features.calculate_summary_features import calculate_summary_stats_number as number_sumstats
from summary_features.calculate_summary_features import calculate_summary_stats_temporal




from utils.simulation_wrapper import event_seed, set_network_default, simulation_wrapper,simulation_wrapper_obs, simulation_wrapper_all

sim_wrapper = simulation_wrapper_all
window_len = 30
prior_min = [0, 0, 0, 0, 17.3, 0, 0, 0, 0, 0, 0, 51.980, 0, 0, 0, 0, 112.13]
prior_max = [0.927, 0.160, 2.093, 0.0519, 35.9, 0.039, 0.000042, 0.854, 0.117, 0.0259, 0.480, 75.08, 0.0000018, 8.633, 0.0537, 4.104, 162.110]


prior = utils.torchutils.BoxUniform(low=prior_min, high=prior_max)

#number_simulations = 10
density_estimator = 'nsf'




from utils import inference
from utils.simulation_wrapper import event_seed, simulation_wrapper
import pickle
from data_load_writer import *
from data_load_writer import load_from_file as lf

import os




import os

print(os.getcwd())

#os.chdir('/home/kathi/Documents/Master_thesis/results_cluster/')



print(os.getcwd())

os.chdir('/home/kathi/Documents/Master_thesis/results_cluster/')

## loading simulations from previously saved computations

file = '10000_sims_17_newparams_default'

file2 = '10000_sims_17_newparams_ERPYes3Trials'
  

thetas = torch.load('{}/step3/thetas.pt'.format(file))
x_without = torch.load('{}/step3/obs_without.pt'.format(file))

thetas2 = torch.load('{}/step3/thetas.pt'.format(file2))
x_without2 = torch.load('{}/step3/obs_without.pt'.format(file2))

x = calculate_summary_stats_temporal(x_without)

x2 = calculate_summary_stats_temporal(x_without2)


density_estimator = 'nsf'



inf = SNPE(prior=prior, density_estimator = density_estimator)

#inf = SNPE_C(prior, density_estimator="nsf")

inf = inf.append_simulations(thetas, x)

density_estimator = inf.train()

posterior = inf.build_posterior(density_estimator)



import pandas as pd

os.chdir('/home/kathi/hnn_out/data/')
trace = pd.read_csv('default/dpl.txt', sep='\t', header=None, dtype= np.float32)
trace_torch = torch.tensor(trace.values, dtype = torch.float32)



obs_real = [torch.index_select(trace_torch, 1, torch.tensor([3])).squeeze(1)]

trace2 = pd.read_csv('ERPYes3Trials/dpl.txt', sep='\t', header=None, dtype= np.float32)
trace_torch2 = torch.tensor(trace.values, dtype = torch.float32)



obs_real2 = [torch.index_select(trace_torch2, 1, torch.tensor([3])).squeeze(1)]