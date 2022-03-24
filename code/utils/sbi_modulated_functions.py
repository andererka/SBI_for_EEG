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

    implements own log_prob() and sample() for two different prior distributions such that parameter sets can be inferred sequentially - one posterior is based on another
   
    takes as arguments:
    - posterior distribution (list) of already inferred parameter set. Can be a list with several posteriors.
    - prior distribution of subsequent parameter set that is dependent on earlier parameter set
    - steps: a list that holds information about how many parameters are inferred in each step.
    """

    has_rsample = False
    support = constraints.real

    def __init__(
        self,
        posterior_distribution,
        prior_distribution,
        validate_args=None,
        batch_shape=torch.Size(),
        event_shape=torch.Size(),
        steps=[0],
    ):
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self._posterior_distribution_list = posterior_distribution
        self._prior_distribution = prior_distribution
        self.steps = steps

        super(Combined, self).__init__(batch_shape, validate_args=validate_args)

        if type(self._posterior_distribution_list) != list:
            self._posterior_distribution_list = [self._posterior_distribution_list]



    def log_prob(self, x):

        """
        calculates the log probability of the combined prior distribution based on some observation x
        """
        steps = self.steps
        
        for i, posterior in enumerate(self._posterior_distribution_list):

            globals()['log_prob_posterior%s' % i] = posterior.log_prob(x[0][steps[i]:steps[i+1]])

            number_rounds = i


        ## for calculating the log probability, we have to add the log probabilities of the single (already inferred) posteriors
        ## with the priors.

        if number_rounds > 0:

            log_prob_posterior_so_far = torch.add(globals()['log_prob_posterior%s' % 0], globals()['log_prob_posterior%s' % 1])

            for i in range(1, number_rounds):
                log_prob_posterior = torch.add(log_prob_posterior_so_far, globals()['log_prob_posterior%s' % int(i+1)])
        else:
            log_prob_posterior = globals()['log_prob_posterior%s' % 0]


        log_prob_prior = self._prior_distribution.log_prob(x[0][steps[number_rounds+1]:])

        log_prob = torch.add(log_prob_posterior, log_prob_prior)


        return log_prob

    def sample(self, sample_shape=torch.Size(), x: Optional[Tensor] = None, show_progress_bars: bool = True, sample_with: Optional[str] = None):

        """
        samples from combined prior distribution

        show_progress_bars and sample_with not used. only needed to conduct sbc
        """



        with torch.no_grad():

            theta_posterior_list = []
            
            for idx, posterior in enumerate(self._posterior_distribution_list):

                if x == None:
                    theta_posterior = posterior.sample(sample_shape)

                else:
                    print('x', x)

                    theta_posterior = posterior.sample(sample_shape, x = x[self.steps[idx]:self.steps[idx+1]])


                #make sure that thetas are in the right shape; otherwise unsqueeze:
                if theta_posterior.dim()  == 1:
                    theta_posterior = torch.unsqueeze(theta_posterior, 0) 

                theta_posterior_list.append(theta_posterior)  




            theta_posterior = torch.cat(tuple(theta_posterior_list), dim = 1)
            theta_prior = self._prior_distribution.sample(sample_shape)

            if theta_prior.dim()  == 1:

                theta_prior = torch.unsqueeze(theta_prior, 0) 

            ### concatenates samples from posterior and prior:
            theta = torch.cat((theta_posterior, theta_prior), 1)

        
            return theta


    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        return torch.mean(self.sample((1000,)),dim = 0)

    @property
    def variance(self):
        """
        Returns the mean of the distribution.
        """
        return torch.var(self.sample((1000,)),dim = 0)





