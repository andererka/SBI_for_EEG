
from utils.simulation_wrapper import set_network_default, simulation_wrapper, simulation_wrapper_extended, simulation_wrapper_obs
from sbi.inference import SNPE_C, prepare_for_sbi, simulate_for_sbi
from summary_features.calculate_summary_features import calculate_summary_stats
from hnn_core import simulate_dipole
import torch
from joblib import Parallel, delayed
from math import sqrt

def run_sim_inference(prior, num_simulations=1000, density_estimator='nsf', num_workers=8):


    #posterior = infer(simulation_wrapper, prior, method='SNPE_C', 
                  #num_simulations=number_simulations, num_workers=4)    
             

    simulator_stats, prior = prepare_for_sbi(simulation_wrapper_obs, prior)

    inference = SNPE_C(prior, density_estimator=density_estimator)

    
    theta, x_without = simulate_for_sbi(simulator_stats, proposal=prior, num_simulations=num_simulations, num_workers=num_workers)

    x = calculate_summary_stats(x_without)

    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator) 


    return posterior, theta, x, x_without

def run_only_inference(theta, x, prior):
    inference = SNPE_C(prior)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train(discard_prior_samples=True)   #discard_prior_samples might speed up training
    posterior = inference.build_posterior(density_estimator) 
    return posterior

def run_only_sim(samples, num_workers=1):
 
    obs_real = Parallel(n_jobs=num_workers, verbose=100, pre_dispatch='1.5*n_jobs', backend='multiprocessing')(delayed(simulation_wrapper_obs)(sample) for sample in samples)
   
    return obs_real

def run_sim_theta_x(prior, num_simulations=1000, density_estimator='nsf', num_workers=8):


    simulator_stats, prior = prepare_for_sbi(simulation_wrapper_obs, prior)

    theta, x_without = simulate_for_sbi(simulator_stats, proposal=prior, num_simulations=num_simulations, num_workers=num_workers)


    return theta, x_without






