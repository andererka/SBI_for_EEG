
import hnn_core
from hnn_core import simulate_dipole, jones_2009_model
from summary_features.calculate_summary_features import calculate_summary_stats, calculate_summary_statistics_alternative
import torch


def simulation_wrapper(params):   #input possibly array of 1 or more params
    """
    Returns summary statistics from conductance values in `params`.

    One can outcomment the line 'net = set_network_2_params' and instead choose 'net = set_network_6_params'
    in order to infer more than 2 parameters

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    if (params.dim()>1):
        param_size = params.size(dim=1)
    else:
        param_size = params.size(dim=0)
 
    if (param_size==2):
        net = set_network_2_params(params)
        print('2 params are investigated')
    elif (param_size==6):
        net = set_network_6_params(params)
        print('6 params are investigated')
    elif (param_size==3):
        net = set_network_3_params(params)
        print('3 params are investigated')
    else:
        print('there is no simulation wrapper defined for this number of parameters!')
        exit()
    
    window_len, scaling_factor = 30, 3000


    dpls = simulate_dipole(net, tstop=170., n_trials=1)
    for dpl in dpls:
        obs = dpl.smooth(window_len).scale(scaling_factor).data['agg']

    #left out summary statistics for a start
    sum_stats= calculate_summary_stats(torch.from_numpy(obs))


    return sum_stats


def simulation_wrapper_extended(params):   #input possibly array of 1 or more params
    """
    Returns summary statistics from conductance values in `params`.

    One can outcomment the line 'net = set_network_2_params' and instead choose 'net = set_network_6_params'
    in order to infer more than 2 parameters

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    if (params.dim()>1):
        param_size = params.size(dim=1)
    else:
        param_size = params.size(dim=0)
 
    if (param_size==2):
        net = set_network_2_params(params)
        print('2 params are investigated')
    elif (param_size==6):
        net = set_network_6_params(params)
        print('6 params are investigated')
    elif (param_size==3):
        net = set_network_3_params(params)
        print('3 params are investigated')
    else:
        print('there is no simulation wrapper defined for this number of parameters!')
        exit()
    
    window_len, scaling_factor = 30, 3000


    dpls = simulate_dipole(net, tstop=170., n_trials=1)
    for dpl in dpls:
        obs = dpl.smooth(window_len).scale(scaling_factor).data['agg']

    #left out summary statistics for a start
    sum_stats= calculate_summary_stats(torch.from_numpy(obs))

    return sum_stats, obs


def simulation_wrapper_obs(params):   #input possibly array of 1 or more params
    """
    Returns summary statistics from conductance values in `params`.

    One can outcomment the line 'net = set_network_2_params' and instead choose 'net = set_network_6_params'
    in order to infer more than 2 parameters

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    if (params.dim()>1):
        param_size = params.size(dim=1)
    else:
        param_size = params.size(dim=0)
 
    if (param_size==2):
        net = set_network_2_params(params)
        print('2 params are investigated')
    elif (param_size==6):
        net = set_network_6_params(params)
        print('6 params are investigated')
    elif (param_size==3):
        net = set_network_3_params(params)
        print('3 params are investigated')
    else:
        print('there is no simulation wrapper defined for this number of parameters!')
        exit()
    
    window_len, scaling_factor = 30, 3000


    dpls = simulate_dipole(net, tstop=170., n_trials=1)
    for dpl in dpls:
        obs = dpl.smooth(window_len).scale(scaling_factor).data['agg']


    return torch.from_numpy(obs)




from random import randrange

def event_seed():
    """
    description: makes sure that one does not take the same random seed for each simulation as it would be the default in the hnn core code;
    permalink to the hnn code location: https://github.com/jonescompneurolab/hnn-core/blob/0406ed1a2b2335b786e83eb1698f27a5c3dcdadc/hnn_core/drives.py#L262

    """
    seed = randrange(2000)
    return seed


def set_network_default(params=None):

    """
    description: sets network to default values for an ERP as described in hnn tutorial
    """

    net = jones_2009_model()
    weights_ampa_d1 = {'L2_basket': 0.006562, 'L2_pyramidal': .000007,
                   'L5_pyramidal': 0.142300}
    weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317,
                   'L5_pyramidal': 0.080074}
    synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
    net.add_evoked_drive(
    'evdist1', mu=63.53, sigma=3.85, numspikes=1, weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1, location='distal',
    synaptic_delays=synaptic_delays_d1, event_seed=event_seed())




    weights_ampa_p1 = {'L2_basket': 0.08831, 'L2_pyramidal': 0.01525,
                   'L5_basket': 0.19934, 'L5_pyramidal': 0.00865}
    synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                        'L5_basket': 1., 'L5_pyramidal': 1.}

    # all NMDA weights are zero; pass None explicitly
    net.add_evoked_drive(
    'evprox1', mu=26.61, sigma=2.47, numspikes=1, weights_ampa=weights_ampa_p1,
    weights_nmda=None, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=event_seed())

    # Second proximal evoked drive. NB: only AMPA weights differ from first
    weights_ampa_p2 = {'L2_basket': 0.000003, 'L2_pyramidal': 1.438840,
                   'L5_basket': 0.008958, 'L5_pyramidal': 0.684013}
    # all NMDA weights are zero; omit weights_nmda (defaults to None)
    net.add_evoked_drive(
    'evprox2', mu=137.12, sigma=8.33, numspikes=1,
    weights_ampa=weights_ampa_p2, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=event_seed())

    return net

def set_network_2_params(params=None):

    """
    description: changes the network due to parameter settings drawn during sbi

    here one changes only the first distal drive and the second proximal drive, which was 
    used as toy example to see if the sbi is working for the case of the hnn simulator.
    """

    net = jones_2009_model()
    weights_ampa_d1 = {'L2_basket': 0.006562, 'L2_pyramidal': .000007,
                   'L5_pyramidal': 0.142300}
    weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317,
                   'L5_pyramidal': 0.080074}
    synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
    net.add_evoked_drive(
    'evdist1', mu=params[0], sigma=3.85, numspikes=1, weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1, location='distal',
    synaptic_delays=synaptic_delays_d1, event_seed=event_seed())




    weights_ampa_p1 = {'L2_basket': 0.08831, 'L2_pyramidal': 0.01525,
                   'L5_basket': 0.19934, 'L5_pyramidal': 0.00865}
    synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                        'L5_basket': 1., 'L5_pyramidal': 1.}

    # all NMDA weights are zero; pass None explicitly
    net.add_evoked_drive(
    'evprox1', mu=26.61, sigma=2.47, numspikes=1, weights_ampa=weights_ampa_p1,
    weights_nmda=None, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=event_seed())

    # Second proximal evoked drive. NB: only AMPA weights differ from first
    weights_ampa_p2 = {'L2_basket': 0.000003, 'L2_pyramidal': 1.438840,
                   'L5_basket': 0.008958, 'L5_pyramidal': 0.684013}
    # all NMDA weights are zero; omit weights_nmda (defaults to None)
    net.add_evoked_drive(
    'evprox2', mu=params[1], sigma=8.33, numspikes=1,
    weights_ampa=weights_ampa_p2, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=event_seed())

    return net


def set_network_3_params(params=None):

    """
    description: changes the network due to parameter settings drawn during sbi

    here one changes only the first distal drive and the second proximal drive, which was 
    used as toy example to see if the sbi is working for the case of the hnn simulator.
    """

    net = jones_2009_model()
    weights_ampa_d1 = {'L2_basket': 0.006562, 'L2_pyramidal': .000007,
                   'L5_pyramidal': 0.142300}
    weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317,
                   'L5_pyramidal': 0.080074}
    synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
    net.add_evoked_drive(
    'evdist1', mu=params[0], sigma=3.85, numspikes=1, weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1, location='distal',
    synaptic_delays=synaptic_delays_d1, event_seed=event_seed())




    weights_ampa_p1 = {'L2_basket': 0.08831, 'L2_pyramidal': 0.01525,
                   'L5_basket': 0.19934, 'L5_pyramidal': 0.00865}
    synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                        'L5_basket': 1., 'L5_pyramidal': 1.}

    # all NMDA weights are zero; pass None explicitly
    net.add_evoked_drive(
    'evprox1', mu=params[1], sigma=2.47, numspikes=1, weights_ampa=weights_ampa_p1,
    weights_nmda=None, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=event_seed())

    # Second proximal evoked drive. NB: only AMPA weights differ from first
    weights_ampa_p2 = {'L2_basket': 0.000003, 'L2_pyramidal': 1.438840,
                   'L5_basket': 0.008958, 'L5_pyramidal': 0.684013}
    # all NMDA weights are zero; omit weights_nmda (defaults to None)
    net.add_evoked_drive(
    'evprox2', mu=params[2], sigma=8.33, numspikes=1,
    weights_ampa=weights_ampa_p2, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=event_seed())

    return net


def set_network_6_params(params=None):

    """
    description: changes the network due to parameter settings drawn during sbi

    """

    net = jones_2009_model()
    weights_ampa_d1 = {'L2_basket': 0.006562, 'L2_pyramidal': .000007,
                   'L5_pyramidal': 0.142300}
    weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317,
                   'L5_pyramidal': 0.080074}
    synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
    net.add_evoked_drive(
    'evdist1', mu=params[0], sigma=params[1], numspikes=1, weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1, location='distal',
    synaptic_delays=synaptic_delays_d1, event_seed=event_seed())




    weights_ampa_p1 = {'L2_basket': 0.08831, 'L2_pyramidal': 0.01525,
                   'L5_basket': 0.19934, 'L5_pyramidal': 0.00865}
    synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                        'L5_basket': 1., 'L5_pyramidal': 1.}

    # all NMDA weights are zero; pass None explicitly
    net.add_evoked_drive(
    'evprox1', mu=params[2], sigma=params[3], numspikes=1, weights_ampa=weights_ampa_p1,
    weights_nmda=None, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=event_seed())

    # Second proximal evoked drive. NB: only AMPA weights differ from first
    weights_ampa_p2 = {'L2_basket': 0.000003, 'L2_pyramidal': 1.438840,
                   'L5_basket': 0.008958, 'L5_pyramidal': 0.684013}
    # all NMDA weights are zero; omit weights_nmda (defaults to None)
    net.add_evoked_drive(
    'evprox2', mu=params[4], sigma=params[5], numspikes=1,
    weights_ampa=weights_ampa_p2, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=event_seed())

    return net 