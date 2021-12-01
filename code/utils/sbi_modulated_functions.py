import torch


from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast


from sbi import utils as utils

import numpy as np

from torch.distributions.distribution import Distribution

from torch.distributions import constraints


from numbers import Number


class Combined(Distribution):
    '''
    Inherits from Torch Distribution class

    implements own log_prob() and sample() for two different prior distributions such that parameter sets can be inferred sequentially - one posterior is based on another
    '''

    has_rsample=False
    support = constraints.real

    def __init__(self, posterior_distribution, prior_distribution, validate_args=None, batch_shape=torch.Size(), event_shape=torch.Size(), number_params_1=0):
        """
        takes as arguments:
        - posterior distribution of already inferred parameter set
        - prior distribution of subsequent parameter set that is dependent on earlier parameter set
        - number_params_1: number of parameters that was inferred already (in posterior distribution)
        """
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self._posterior_distribution = posterior_distribution
        self._prior_distribution = prior_distribution
        self.number_params_1 = number_params_1

        super(Combined, self).__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, x):

        """
        calculates the log probability of the combined prior distribution based on some observation x
        """
        index = self.number_params_1
   
        log_prob_posterior = self._posterior_distribution.log_prob(x[0][:index])
        log_prob_prior = self._prior_distribution.log_prob(x[0][index:])

        log_prob = torch.add(log_prob_posterior, log_prob_prior)


        print(log_prob)
        #print(torch.unsqueeze(log_prob, 0))

        return log_prob

    def sample(self, sample_shape=torch.Size()):

        """
        samples from combined prior distribution
        """

        with torch.no_grad():
            theta_posterior = self._posterior_distribution.sample(sample_shape)
            theta_prior = self._prior_distribution.sample(sample_shape)
            print('theta pos size', theta_posterior.size())
            print('theta prior size', theta_prior.size())
            print(theta_posterior.dim())
            if (theta_posterior.dim()==1):
                print('true')
                theta_posterior = torch.unsqueeze(theta_posterior, 0)
                theta_prior = torch.unsqueeze(theta_prior, 0)

          

            theta = torch.cat((theta_posterior, theta_prior), 1)
            print('theta', theta)
            return theta
