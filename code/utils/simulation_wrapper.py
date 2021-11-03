
import hnn_core
from hnn_core import simulate_dipole, jones_2009_model
from summary_features.calculate_summary_features import calculate_summary_stats, calculate_summary_statistics_alternative
import torch

net = jones_2009_model()

def simulation_wrapper(params):   #input possibly array of 1 or more params
    """
    Returns summary statistics from conductance values in `params`.

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    
    
    window_len, scaling_factor = 30, 3000
    net._params['t_evdist_1'] = params[0]
    #net._params['sigma_t_evdist_1'] = params[1]
    net._params['t_evprox_2'] = params[1]
    #net._params['sigma_t_evprox_2'] = params[3]
    #net._params['L2Pyr_soma_diam'] = params[4]   # 23.4
    #net._params['L5Pyr_apical1_diam'] = params[5]     # 7.48

    ##simulates 8 trials at a time like this
    dpls = simulate_dipole(net, tstop=170., n_trials=1)
    for dpl in dpls:
        obs = dpl.smooth(window_len).scale(scaling_factor).data['agg']

    #left out summary statistics for a start
    sum_stats = calculate_summary_stats(torch.from_numpy(obs))
    return sum_stats
