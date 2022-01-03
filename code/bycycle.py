from bycycle import features
from utils.simulation_wrapper import (
    simulation_wrapper,
    simulation_wrapper_obs,

)
import torch

import matplotlib.pyplot as plt



from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE_C, prepare_for_sbi, simulate_for_sbi

from utils import inference


fs = 1000
f_range = (4, 10)

sim_wrapper = simulation_wrapper_obs
prior_min = [7.9, 43.8] 

prior_max = [30, 79.9]

true_params = torch.tensor([[26.61, 63.53]])

prior = utils.torchutils.BoxUniform(low=prior_min, high=prior_max)

theta, x_without = inference.run_sim_theta_x(
        prior = prior,
        simulation_wrapper= sim_wrapper,
        num_simulations=1,
        num_workers=1,
    )

plt.plot(x_without)

df_features = features.compute_features(x_without, fs, f_range)


