import torch

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from sbi import utils as utils

import numpy as np

from torch.distributions.distribution import Distribution

from sbi.inference.posteriors.base_posterior import NeuralPosterior

from torch.distributions import constraints

from torch import Tensor


class Combined(Distribution):
    """
    Inherits from Torch Distribution class
    - This class implements a distribution that combines a prior distribution with a posterior distribution. Therefore, one has to specify the number of parameters that was already inferred (number_params_1)
    
    - implements own log_prob() and sample() for two different prior distributions such that parameter sets can be inferred sequentially

    - for further explanation, please have a look at the thesis folder: The method section of my thesis explains the idea behind it.
   
    takes as arguments:
    - posterior distribution of already inferred parameter set
    - prior distribution of subsequent parameter set that is dependent on earlier parameter set
    - number_params_1: number of parameters that was inferred already (in posterior distribution)
    """

    has_rsample = False
    support = constraints.real

    def __init__(
        self,
        posterior_distribution,
        prior_distribution,
        validate_args={},
        batch_shape=torch.Size(),
        event_shape=torch.Size(),
        number_params_1=0,
    ):
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self._posterior_distribution = posterior_distribution
        self._prior_distribution = prior_distribution
        self.number_params = number_params_1

        super(Combined, self).__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, x):

        """
        calculates the log probability of the combined prior distribution based on some observation x
        """
        index = self.number_params

        print("x shape", x.shape)
        print("index", index)

        log_prob_posterior = self._posterior_distribution.log_prob(x[0][:index])
        log_prob_prior = self._prior_distribution.log_prob(x[0][index:])

        log_prob = torch.add(log_prob_posterior, log_prob_prior)

        print("log prob", log_prob)

        return log_prob

    def sample(
        self,
        sample_shape=torch.Size(),
        x: Optional[Tensor] = None,
        show_progress_bars: bool = True,
        sample_with: Optional[str] = None,
    ):

        """
        samples from combined prior distribution
        """

        with torch.no_grad():

            if x == None:
                theta_posterior = self._posterior_distribution.sample(sample_shape)

            else:

                theta_posterior = self._posterior_distribution.sample(
                    sample_shape, x=x[:, : x.shape[1]]
                )

            print("theta posterior shape", theta_posterior.shape)

            theta_prior = self._prior_distribution.sample(sample_shape)

            # make sure that thetas are in the right shape; otherwise unsqueeze:
            if theta_posterior.dim() == 1:

                theta_posterior = torch.unsqueeze(theta_posterior, 0)

            if theta_prior.dim() == 1:

                theta_prior = torch.unsqueeze(theta_prior, 0)

            theta = torch.cat((theta_posterior, theta_prior), 1)

            print("theta shape", theta.shape)

            return theta

    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        return torch.mean(self.sample((1000,)), dim=0)

    @property
    def variance(self):
        """
        Returns the mean of the distribution.
        """
        return torch.var(self.sample((1000,)), dim=0)
