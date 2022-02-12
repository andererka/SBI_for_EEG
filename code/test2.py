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

from utils.simulation_wrapper import SimulationWrapper, simulation_wrapper_all, simulation_wrapper_obs
from summary_features.calculate_summary_features import (
    calculate_summary_stats_temporal

)

true_params = torch.tensor([[0.277, 0.0399, 0.6244, 0.3739, 0.0, 18.977, 0.000012, 0.0115, 0.0134,  0.0767, 0.06337, 63.08, 4.6729, 2.33, 0.016733, 0.0679, 120.86]])



print(true_params.shape[0])


