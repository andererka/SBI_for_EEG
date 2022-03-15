#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt


# sbi
from sbi import utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi

from utils import inference

from utils.simulation_wrapper import SimulationWrapper
from summary_features.calculate_summary_features import (
    calculate_summary_stats_temporal

)

sim_wrapper = SimulationWrapper(num_params=25, small_steps=True)

true_params = torch.tensor([[0.034, 0.0, 0.6244, 0.3739, 0.0399, 0.0, 0.277, 0.3739, 18.977, 
                0.466095, 0.0767, 0.000012, 0.013407, 0.011467, 0.06337, 63.08, 
                2.33, 0.0679, 0.011468, 0.061556, 4.6729, 0.016733, 0.000005, 0.116706, 120.86]])



prior_min = [0, 0, 0, 0, 0, 0, 0, 0, 17.3,    # prox1 weights
            0, 0, 0, 0, 0, 0, 51.980,            # distal weights
            0, 0, 0, 0, 0, 0, 0, 0, 112.13]       # prox2 weights



prior_max = [0.927, 1.0, 0.160, 1.0,  2.093, 1.0, 0.0519, 1.0, 35.9,
            0.0394, 0.117, 0.000042, 0.025902, 0.854, 0.480, 75.08, 
            0.000018, 1.0, 8.633, 1.0, 0.05375, 1.0, 4.104,  1.0, 162.110]





prior = utils.torchutils.BoxUniform(low=prior_min, high=prior_max)


obs_real_complete = inference.run_only_sim(
torch.tensor([list(true_params[0][0:])]), 
simulation_wrapper = sim_wrapper, 
num_workers=1
)

print(obs_real_complete.shape)

obs_real = obs_real_complete[0][:2800]


obs_real_stat = calculate_summary_stats_temporal(obs_real)

print(obs_real_stat[0])