
from utils.simulation_wrapper import set_network_default
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from summary_features.calculate_summary_features import calculate_summary_stats
from hnn_core import simulate_dipole
import torch
from joblib import Parallel, delayed
from math import sqrt

def run_sim_inference(prior, simulation_wrapper, num_simulations=1000, density_estimator='nsf', num_workers=8):


    #posterior = infer(simulation_wrapper, prior, method='SNPE_C', 
                  #num_simulations=number_simulations, num_workers=4)    
             

    simulator, prior = prepare_for_sbi(simulation_wrapper, prior)
    inference = SNPE(prior, density_estimator=density_estimator)


    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_simulations, num_workers=num_workers)

    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator) 

    return posterior, theta, x

def run_only_inference(theta, x, prior):
    inference = SNPE(prior)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator) 
    return posterior

def run_only_sim(samples, num_workers=1):
 
    print('samples', samples)
    for sample in samples:
        print('sample', sample)
    s_x = Parallel(n_jobs=num_workers, verbose=100, pre_dispatch='1.5*n_jobs', backend='multiprocessing')(delayed(simulation_util)(sample) for sample in samples)
    print('done')
    return s_x


def simulation_util(sample):

    net = set_network_default()

    window_len = 30
    scaling_factor = 3000
    print('sample0',sample)
    net._params['t_evdist_1'] = int(sample[0])
    #net._params['sigma_t_evdist_1'] = 3.85
    net._params['t_evdist_2'] = int(sample[1])
    #net._params['sigma_t_evprox_2'] = 8.33

    dpls = simulate_dipole(net, tstop=170., n_trials=1)
    for dpl in dpls:
        obs = dpl.smooth(window_len).scale(scaling_factor).data['agg']

    s_x = calculate_summary_stats(torch.from_numpy(obs))
    print('done1')
    return s_x