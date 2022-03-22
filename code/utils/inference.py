from utils.simulation_wrapper import (
    SimulationWrapper
)

from sbi.inference import SNPE_C, prepare_for_sbi, simulate_for_sbi
from summary_features.calculate_summary_features import calculate_summary_stats_number, calculate_summary_stats_temporal
from hnn_core import simulate_dipole
import torch
from joblib import Parallel, delayed
from math import sqrt


sim_wrapper = SimulationWrapper()



def run_only_sim(samples, simulation_wrapper=sim_wrapper, num_workers=1):

    obs_real = Parallel(
        n_jobs=num_workers,
        verbose=100,
        pre_dispatch="1.5*n_jobs",
        backend="multiprocessing",
    )(delayed(simulation_wrapper)(sample) for sample in samples)

    return obs_real


def run_sim_theta_x(
    prior, simulation_wrapper=sim_wrapper, num_simulations=1000, num_workers=8):

    simulator_stats, prior = prepare_for_sbi(simulation_wrapper, prior)


    theta, x_without = simulate_for_sbi(
        simulator_stats,
        proposal=prior,
        num_simulations=num_simulations,
        num_workers=num_workers,
        show_progress_bar=False
    
    )

    return theta, x_without
