from data_load_writer import load_from_file as lf
from data_load_writer import write_to_file
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
import os

print(os.getcwd())
import pickle




### loading the simulated data:
with open('results/ERP_results11-02-2021_15:37:00/class', 'rb') as pickle_file:
    file_writer = pickle.load(pickle_file)


prior = lf.load_prior(file_writer.folder)
thetas = lf.load_thetas(file_writer.folder)
x = lf.load_obs(file_writer.folder)

true_params = lf.load_true_params(file_writer.folder)
print(thetas.shape)


print(type(thetas))

from utils import inference

posterior = inference.run_only_inference(theta=thetas, x=x, prior=prior)



# set params as the 'true parameters'
net._params['t_evdist_1'] = 63.53
net._params['sigma_t_evdist_1'] = 3.85
net._params['t_evdist_2'] = 137.12
net._params['sigma_t_evprox_2'] = 8.33


window_len = 30
scaling_factor = 3000
dpls = simulate_dipole(net, tstop=170., n_trials=4)
for dpl in dpls:
    obs = dpl.smooth(window_len).scale(scaling_factor).data['agg']

obs_real = calculate_summary_stats(torch.from_numpy(obs))

samples = posterior.sample((100,), 
                           x=obs_real)




true_params = torch.Tensor([63.53, 3.85, 137.12, 8.33])




fig, axes = analysis.pairplot(samples,
                           #limits=[[.5,80], [1e-4,15.]],
                           #ticks=[[.5,80], [1e-4,15.]],
                           figsize=(5,5),
                           points=true_params,
                           points_offdiag={'markersize': 6},
                           points_colors='r');


file_writer = write_to_file.WriteToFile(experiment='ERP_inference_only')

file_writer.save_all(posterior, prior, theta, x, fig)
