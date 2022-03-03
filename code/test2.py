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

import torch
xx = torch.zeros(3, 4)
xx.foo = 'bar'
torch.save(xx, '_xx.pt')
torch.load('_xx.pt').foo