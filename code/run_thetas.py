"""
samples prior thetas and simulates observations xs for these (in order to run on slurm and afterwards 
use for SBC)
"""

import matplotlib.pyplot as plt


import sys

sys.path.append("../code/")


import numpy as np
import torch

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt


import os


import utils.sbi_modulated_functions

# sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE


from utils import inference

from utils.simulation_wrapper import SimulationWrapper


num_params = 17
sim_wrapper = SimulationWrapper(num_params)


window_len = 30

if num_params == 25:
    prior_min = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        13.3,  # prox1 weights
        0,
        0,
        0,
        0,
        0,
        0,
        51.980,  # distal weights
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        112.13,
    ]  # prox2 weights

    # ampa, nmda [0.927, 0.160, 2.093, 0.0519,        1.0, 1.0, 1.0, 1.0, 35.9,
    #           0.0394, 0.000042, 0.039372,           0.854, 0.117,  0.480, 75.08,
    #            0.000018, 8.633, 0.05375, 4.104,     1.0, 1.0, 1.0, 1.0, 162.110]

    prior_max = [
        0.927,
        1.0,
        0.160,
        1.0,
        2.093,
        1.0,
        0.0519,
        1.0,
        35.9,
        0.0394,
        0.117,
        0.000042,
        0.025902,
        0.854,
        0.480,
        75.08,
        0.000018,
        1.0,
        8.633,
        1.0,
        0.05375,
        1.0,
        4.104,
        1.0,
        162.110,
    ]

    true_params = torch.tensor(
        [
            [
                0.277,
                0.3739,
                0.0399,
                0.0,
                0.6244,
                0.3739,
                0.034,
                0.0,
                18.977,
                0.011467,
                0.06337,
                0.000012,
                0.013407,
                0.466095,
                0.0767,
                63.08,
                0.000005,
                0.116706,
                4.6729,
                0.016733,
                0.011468,
                0.061556,
                2.33,
                0.0679,
                120.86,
            ]
        ]
    )

if num_params == 17:

    prior_min = [0, 0, 0, 0, 0, 13.3, 0, 0, 0, 0, 0, 51.980, 0, 0, 0, 0, 112.13]
    prior_max = [
        0.927,
        0.160,
        2.093,
        1.0,
        1.0,
        35.9,
        0.000042,
        0.039372,
        0.025902,
        0.480,
        0.117,
        75.08,
        8.633,
        4.104,
        1.0,
        1.0,
        162.110,
    ]

    true_params = torch.tensor(
        [
            [
                0.277,
                0.0399,
                0.6244,
                0.3739,
                0.0,
                18.977,
                0.000012,
                0.0115,
                0.0134,
                0.0767,
                0.06337,
                63.08,
                4.6729,
                2.33,
                0.016733,
                0.0679,
                120.86,
            ]
        ]
    )


prior = utils.torchutils.BoxUniform(low=prior_min, high=prior_max)


# generate ground truth parameters and corresponding simulated observations for SBC.
thetas = prior.sample((1000,))
xs = inference.run_only_sim(thetas, sim_wrapper, num_workers=60)
torch.save(xs, "1000xs.pt")
torch.save(thetas, "1000thetas_prior.pt")
