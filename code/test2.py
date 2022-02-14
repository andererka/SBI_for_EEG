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

from utils import inference

from utils.simulation_wrapper import SimulationWrapper
from summary_features.calculate_summary_features import (
    calculate_summary_stats_temporal

)

sim_wrapper = SimulationWrapper(num_params=25)

true_params = torch.tensor([[0.034, 0.0, 0.6244, 0.3739, 0.0399, 0.0, 0.277, 0.3739, 18.977, 
                0.466095, 0.0767, 0.000012, 0.013407, 0.011467, 0.06337, 63.08, 
                2.33, 0.0679, 0.011468, 0.061556, 4.6729, 0.016733, 0.000005, 0.116706, 120.86]])


obs_real = inference.run_only_sim(
        true_params, 
        simulation_wrapper = sim_wrapper, 
        num_workers=1
    ) 


