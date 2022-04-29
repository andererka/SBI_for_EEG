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

        print('x shape', x.shape)
        print('index', index)

        log_prob_posterior = self._posterior_distribution.log_prob(x[0][:index])
        log_prob_prior = self._prior_distribution.log_prob(x[0][index:])

        log_prob = torch.add(log_prob_posterior, log_prob_prior)

        print('log prob', log_prob)


        return log_prob

    def sample(self, sample_shape=torch.Size(), x: Optional[Tensor] = None, show_progress_bars: bool = True, sample_with: Optional[str] = None):

        """
        samples from combined prior distribution
        """

        with torch.no_grad():

            if x == None:
                theta_posterior = self._posterior_distribution.sample(sample_shape)

            else:

                theta_posterior = self._posterior_distribution.sample(sample_shape, x = x[:,:x.shape[1]])

            print('theta posterior shape', theta_posterior.shape)

            theta_prior = self._prior_distribution.sample(sample_shape)
    
            #make sure that thetas are in the right shape; otherwise unsqueeze:
            if theta_posterior.dim() == 1:
                
                theta_posterior = torch.unsqueeze(theta_posterior, 0)

            if theta_prior.dim() == 1:

                theta_prior = torch.unsqueeze(theta_prior, 0)


            theta = torch.cat((theta_posterior, theta_prior), 1)

            print('theta shape', theta.shape)


        
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



class Combined2(Distribution):
    """
    Inherits from Torch Distribution class

    implements own log_prob() and sample() for two different prior distributions such that parameter sets can be inferred sequentially - one posterior is based on another
   
    takes as arguments:
    - posterior distribution (list) of already inferred parameter set. Can be a list with several posteriors.
    - prior distribution of subsequent parameter set that is dependent on earlier parameter set
    - steps: a list that holds information about how many parameters are inferred in each step. Default is [0].
    - validate_args, batch_shape and event_shape as in DirectPosterior class from sbi package

    has the following functions:
    - log_prob: calculates the log probability according to list of posteriors and prior (that all are allocated to
    a specific number of thetas - acording to steps.)
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
        steps=[0],
    ):
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self._posterior_distribution_list = posterior_distribution
        self._prior_distribution = prior_distribution
        self.steps = steps

        try:

            self.default_x_list = [self._posterior_distribution_list[i].default_x for i in range(len(self._posterior_distribution_list))]
            self._x_shape_list = [self._posterior_distribution_list[i].default_x.shape for i in range(len(self._posterior_distribution_list))]
            self._x_shape = self._posterior_distribution_list[len(self._posterior_distribution_list)].default_x.shape
            self.default_x = self._posterior_distribution_list[len(self._posterior_distribution_list)].default_x 

        except:
            self.default_x = None
        

        super(Combined2, self).__init__(batch_shape, validate_args=validate_args)

        if type(self._posterior_distribution_list) != list:
            self._posterior_distribution_list = [self._posterior_distribution_list]





    def log_prob(self, x):

        """
        calculates the log probability of the combined prior distribution based on some observation x
        """

        steps = self.steps
        
        for i, posterior in enumerate(self._posterior_distribution_list):

            print(steps[i], steps[i+1])


            globals()['log_prob_posterior%s' % i] = posterior.log_prob(x[0][steps[i]:steps[i+1]])

            number_rounds = i


        ## for calculating the log probability, we have to add the log probabilities of the single (already inferred) posteriors
        ## with the priors.

        if number_rounds == 0:
            log_prob_posterior = globals()['log_prob_posterior%s' % 0]

        if number_rounds > 0:

            log_prob_posterior_so_far = torch.add(globals()['log_prob_posterior%s' % 0], globals()['log_prob_posterior%s' % 1])

            if number_rounds == 2:
                    log_prob_posterior_so_far = torch.add(log_prob_posterior_so_far, globals()['log_prob_posterior%s' % 2])

            log_prob_posterior = log_prob_posterior_so_far
            
        print('log prob posterior', log_prob_posterior)
            

        if self._prior_distribution != None:
            log_prob_prior = self._prior_distribution.log_prob(x[0][steps[number_rounds+1]:])

            log_prob = torch.add(log_prob_posterior, log_prob_prior)
        else:
            log_prob = log_prob_posterior


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
                    ## in earlier rounds not all of the summary statistics are used such
                    ## that we need this indix:
                    x_shape = self._x_shape_list[idx]

                    theta_posterior = posterior.sample(sample_shape, x = x[:,:x_shape[1]])

                    print('theta posterior shape', theta_posterior.shape)


                #make sure that thetas are in the right shape; otherwise unsqueeze:
                if theta_posterior.dim()  == 1:
                    theta_posterior = torch.unsqueeze(theta_posterior, 0) 

                theta_posterior_list.append(theta_posterior)  



            ## concatenates the thetas from the different posteriors together
            theta_posterior = torch.cat(tuple(theta_posterior_list), dim = 1)

            if self._prior_distribution != None:
                theta_prior = self._prior_distribution.sample(sample_shape)

                if theta_prior.dim()  == 1:

                    theta_prior = torch.unsqueeze(theta_prior, 0) 

                ### concatenates thetas from posterior and prior:
                theta = torch.cat((theta_posterior, theta_prior), 1)

                print('theta shape', theta.shape)
            else:
                theta = theta_posterior

        
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


class Combine_List(Distribution):
    """
    Inherits from Torch Distribution class
    implements own log_prob() and sample() 
    takes as arguments:
    - posterior list with posteriors from separate steps
    - steps: specifies how many thetas should be taken from the separate priors
    """

    has_rsample = False
    support = constraints.real

    def __init__(
        self,
        posterior_list,
        steps = [0],
        validate_args={},
        batch_shape=torch.Size(),
        event_shape=torch.Size(),
    ):
        self._batch_shape = batch_shape
        self._event_shape = event_shape

        self.posterior_list = posterior_list

        self.steps = steps

        super(Combine_List, self).__init__(batch_shape, validate_args=validate_args)





    def log_prob(self, x):

        """
        calculates the log probability of the combined prior distribution based on some observation x
        """

        steps = self.steps

        log_prob_posterior = 0
        
        for i, posterior in enumerate(self.posterior_list):


            globals()['log_prob_posterior%s' % i] = posterior.log_prob(x[0][steps[i]:steps[i+1]])

            log_prob_posterior += globals()['log_prob_posterior%s' % i]


        return log_prob_posterior


    def sample(self, sample_shape=torch.Size(), x: Optional[Tensor] = None, show_progress_bars: bool = True, sample_with: Optional[str] = None):

        """
        samples from combined prior distribution

        show_progress_bars and sample_with not used. only needed to conduct sbc
        """



        with torch.no_grad():

            theta_posterior_list = []
            
            for idx, posterior in enumerate(self.posterior_list):

                if x == None:
                    theta_posterior = posterior.sample(sample_shape)

                else:
                    ## in earlier rounds not all of the summary statistics are used such
                    ## that we need this indix:
                    x_shape = self._x_shape_list[idx]

                    theta_posterior = posterior.sample(sample_shape, x = x[:,:x_shape[1]])


                #make sure that thetas are in the right shape; otherwise unsqueeze:
                if theta_posterior.dim()  == 1:
                    theta_posterior = torch.unsqueeze(theta_posterior, 0) 

                ## select only the subset that was last inferred in a particular posterior:
                theta_posterior = theta_posterior[:,self.steps[idx]:self.steps[idx+1]]

                print('theta posteriorrr', theta_posterior, theta_posterior.shape)

                theta_posterior_list.append(theta_posterior)  



            ## concatenates the thetas from the different posteriors together
            theta_posterior = torch.cat(tuple(theta_posterior_list), dim = 1)

        
            return theta_posterior

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















