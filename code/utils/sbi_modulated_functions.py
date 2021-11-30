import torch

from utils.simulation_wrapper import (
    set_network_default,
    simulation_wrapper,
    simulation_wrapper_extended,
    simulation_wrapper_obs,
)

from sbi.inference.base import check_if_proposal_has_default_x
from sbi.simulators.simutils import simulate_in_batches

from torch import Tensor


from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast


import numpy as np

prior_min = [43.8, 7.9, 89.49]  # 't_evdist_1', 't_evprox_1', 't_evprox_2'

prior_max = [79.9, 30, 152.96]

true_params = torch.tensor([[63.53, 26.61, 137.12]])
parameter_names = ["t_evdist_1", "t_evprox_1", "t_evprox_2"]


def simulate_for_sbi_sequential(
    simulator: Callable,
    proposal: Any,
    num_simulations: int,
    num_workers: int = 1,
    simulation_batch_size: int = 1,
    show_progress_bar: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""
    Returns ($\theta, x$) pairs obtained from sampling the proposal and simulating.
    This function performs two steps:
    - Sample parameters $\theta$ from the `proposal`.
    - Simulate these parameters to obtain $x$.
    Args:
        simulator: A function that takes parameters $\theta$ and maps them to
            simulations, or observations, `x`, $\text{sim}(\theta)\to x$. Any
            regular Python callable (i.e. function or class with `__call__` method)
            can be used.
        proposal: Probability distribution that the parameters $\theta$ are sampled
            from.
        num_simulations: Number of simulations that are run.
        num_workers: Number of parallel workers to use for simulations.
        simulation_batch_size: Number of parameter sets that the simulator
            maps to data x at once. If None, we simulate all parameter sets at the
            same time. If >= 1, the simulator has to process data of shape
            (simulation_batch_size, parameter_dimension).
        show_progress_bar: Whether to show a progress bar for simulating. This will not
            affect whether there will be a progressbar while drawing samples from the
            proposal.
    Returns: Sampled parameters $\theta$ and simulation-outputs $x$.
    """
    theta_all = []
    for prior in proposal:
        check_if_proposal_has_default_x(prior)

        theta = prior.sample((num_simulations,))

        theta_all.append(theta)

    theta = torch.cat((theta_all), 1)

    x = simulate_in_batches(
        simulator, theta, simulation_batch_size, num_workers, show_progress_bar
    )

    return theta, x


import torch
from scipy.stats import moment
from scipy.stats import norm
from torch.distributions.bernoulli import Bernoulli

from utils.simulation_wrapper import (
    set_network_default,
    simulation_wrapper,
    simulation_wrapper_extended,
    simulation_wrapper_obs,
)


import sbi.inference
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.simulators.simutils import simulate_in_batches

from torch import Tensor


from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast


from sbi.inference.base import check_if_proposal_has_default_x

from sbi import utils as utils

from utils.inference import simulate_for_sbi
import numpy as np

prior_min = [43.8, 7.9, 89.49]  # 't_evdist_1', 't_evprox_1', 't_evprox_2'

prior_max = [79.9, 30, 152.96]

true_params = torch.tensor([[63.53, 26.61, 137.12]])
parameter_names = ["t_evdist_1", "t_evprox_1", "t_evprox_2"]

from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


from numbers import Number


class Combined(Distribution):
    def __init__(self, posterior_distribution, prior_distribution, validate_args=None):

        self._posterior_distribution = posterior_distribution
        self._prior_distribution = prior_distribution

    def log_prob(self, x):
        log_prob_posterior = self._posterior_distribution.log_prob(x)
        log_prob_prior = self._prior_distribution.log.prob(x)
        log_prob = torch.cat(([log_prob_posterior, log_prob_prior]), 1)
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            theta_posterior = self._posterior_distribution.sample(sample_shape)
            theta_prior = self._prior_distribution.sample(sample_shape)

        theta = torch.cat(([theta_posterior, theta_prior]), 1)
        return theta
