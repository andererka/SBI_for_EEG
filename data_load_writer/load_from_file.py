import sys, os, datetime
import torch
import json
import datetime



    
def load_posterior(path_to_file):

    posterior = torch.load(path_to_file+'/posterior.pt')
    return posterior

def load_prior(path_to_file):

    prior = torch.load(path_to_file+ '/prior.pt')
    return prior

def load_obs(path_to_file):

    obs = torch.load(path_to_file+ '/obs.pt')
    return obs


def load_thetas(path_to_file):

    thetas = torch.load(path_to_file+ '/thetas.pt')
    return thetas

def load_true_params(path_to_file):

    params = torch.load(path_to_file+ '/true_params.pt')
    return params
