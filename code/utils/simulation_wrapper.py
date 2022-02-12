import hnn_core
from hnn_core import simulate_dipole, jones_2009_model
from summary_features.calculate_summary_features import (
    calculate_summary_stats_number
)
import torch


class SimulationWrapper:
    def __init__(self, num_params = 17, change_order = False, incremental = True, only_one = False):
        self.num_params = num_params
        self.change_order = change_order
        self.incremental = incremental
        self.only_one = only_one
    

    def __call__(self, params):
        if (self.num_params == 17 or self.num_params == 6) and (self.incremental == True):
            if (self.change_order == False) and (self.only_one == False):
                return simulation_wrapper_all(params)
            elif (self.change_order == False) and (self.only_one == True):
                return simulation_wrapper_all_only_one(params)       



def simulation_wrapper_all(params):  # input possibly array of 1 or more params
    """
    Returns summary statistics from conductance values in `params`.

    One can outcomment the line 'net = set_network_2_params' and instead choose 'net = set_network_6_params'
    in order to infer more than 2 parameters

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """

    early_stop = 200.0


    if params.dim() == 1:
        param_size = params.size(dim=0)
    else:
        param_size = params.size(dim=1)

    print('param size', param_size)


    if (param_size == 6):
        
        early_stop = 70.0
        print('6 params investigated')

    if (param_size == 2):
        
        early_stop = 70.0
        print('2 params investigated')

    if (param_size == 12):
        print('12 params investigated')

        early_stop = 120.0

    if (param_size == 4):
        print('4 params investigated')

        early_stop = 120.0
    
    print('early stop', early_stop)
    print('param size ', param_size)

    params = params.tolist()
 
    net = set_network_weights(params)

    window_len, scaling_factor = 30, 3000

    dpls = simulate_dipole(net, tstop=early_stop, n_trials=1)
    for dpl in dpls:
        obs = dpl.smooth(window_len).scale(scaling_factor).data["agg"]
        #obs = dpl.smooth(window_len).data["agg"]

    return torch.from_numpy(obs)


def simulation_wrapper_all_only_one(params):  # input possibly array of 1 or more params
    """
    Returns summary statistics from conductance values in `params`.

    One can outcomment the line 'net = set_network_2_params' and instead choose 'net = set_network_6_params'
    in order to infer more than 2 parameters

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """

    early_stop = 200.0


    if params.dim() == 1:
        param_size = params.size(dim=0)
    else:
        param_size = params.size(dim=1)

    print('param size', param_size)


    if (param_size == 6):
        
        early_stop = 70.0
        print('6 params investigated')

    if (param_size == 2):
        
        early_stop = 70.0
        print('2 params investigated')

    if (param_size == 12):
        print('12 params investigated')

        early_stop = 120.0

    if (param_size == 4):
        print('4 params investigated')

        early_stop = 120.0
    
    print('early stop', early_stop)
    print('param size ', param_size)

    params = params.tolist()
 
    net = set_network_weights_only_one(params)

    window_len, scaling_factor = 30, 3000

    dpls = simulate_dipole(net, tstop=early_stop, n_trials=1)
    for dpl in dpls:
        obs = dpl.smooth(window_len).scale(scaling_factor).data["agg"]
        #obs = dpl.smooth(window_len).data["agg"]

    return torch.from_numpy(obs)





def simulation_wrapper(params):  # input possibly array of 1 or more params
    """
    Returns summary statistics from conductance values in `params`.

    One can outcomment the line 'net = set_network_2_params' and instead choose 'net = set_network_6_params'
    in order to infer more than 2 parameters

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    if params.dim() > 1:
        param_size = params.size(dim=1)
    else:
        param_size = params.size(dim=0)

    if param_size == 1:
        net = set_network_1_params(params)
        print("1 params are investigated")

    if param_size == 2:
        net = set_network_2_params(params)
        print("2 params are investigated")

    elif param_size == 3:
        net = set_network_3_params(params)
        print("3 params are investigated")
    else:
        print(
            "there is no simulation wrapper defined for this number of parameters! kkk"
        )
        print("param size", param_size)

    window_len, scaling_factor = 30, 3000

    dpls = simulate_dipole(net, tstop=170.0, n_trials=1)
    for dpl in dpls:
        obs = dpl.smooth(window_len).scale(scaling_factor).data["agg"]

    # left out summary statistics for a start
    sum_stats = calculate_summary_stats_number(torch.from_numpy(obs))

    return sum_stats


def simulation_wrapper_extended(params):  # input possibly array of 1 or more params
    """
    Returns summary statistics from conductance values in `params`.

    One can outcomment the line 'net = set_network_2_params' and instead choose 'net = set_network_6_params'
    in order to infer more than 2 parameters

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    if params.dim() > 1:
        param_size = params.size(dim=1)
    else:
        param_size = params.size(dim=0)

    if param_size == 2:
        net = set_network_2_params(params)
        print("2 params are investigated")

    elif param_size == 3:
        net = set_network_3_params(params)
        print("3 params are investigated")
    else:
        print("there is no simulation wrapper defined for this number of parameters!")
        exit()

    window_len, scaling_factor = 30, 3000

    dpls = simulate_dipole(net, tstop=170.0, n_trials=1)
    for dpl in dpls:
        obs = dpl.smooth(window_len).scale(scaling_factor).data["agg"]

    # left out summary statistics for a start
    sum_stats = calculate_summary_stats_number(torch.from_numpy(obs))

    return sum_stats, obs


def simulation_wrapper_obs(params):  # input possibly array of 1 or more params
    """
    Returns summary statistics from conductance values in `params`.

    One can outcomment the line 'net = set_network_2_params' and instead choose 'net = set_network_6_params'
    in order to infer more than 2 parameters

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """

    early_stop = 170.0


    if params.dim() > 1:
        param_size = params.size(dim=1)
    else:
        param_size = params.size(dim=0)

    print(param_size)
    print("params size", params.size())

    if param_size == 1:
        net = set_network_1_params(params)
        print("1 params are investigated")

        #early_stop = 90.0

    elif param_size == 2:
        net = set_network_2_params(params)
        print("2 params are investigated")

        #early_stop = 140.0

    elif param_size == 3:
        net = set_network_3_params(params)
        print("3 params are investigated")
        
    else:
        print("there is no simulation wrapper defined for this number of parameters!")
        exit()

    window_len, scaling_factor = 30, 3000

    dpls = simulate_dipole(net, tstop=early_stop, n_trials=1)
    for dpl in dpls:
        obs = dpl.smooth(window_len).scale(scaling_factor).data["agg"]

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
    weights_ampa_d1 = {
        "L2_basket": 0.006562,
        "L2_pyramidal": 0.000007,
        "L5_pyramidal": 0.142300,
    }
    weights_nmda_d1 = {
        "L2_basket": 0.019482,
        "L2_pyramidal": 0.004317,
        "L5_pyramidal": 0.080074,
    }
    synaptic_delays_d1 = {"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1}
    net.add_evoked_drive(
        "evdist1",
        mu=63.53,
        sigma=3.85,
        numspikes=1,
        weights_ampa=weights_ampa_d1,
        weights_nmda=weights_nmda_d1,
        location="distal",
        synaptic_delays=synaptic_delays_d1,
        event_seed=event_seed(),
    )

    weights_ampa_p1 = {
        "L2_basket": 0.08831,
        "L2_pyramidal": 0.01525,
        "L5_basket": 0.19934,
        "L5_pyramidal": 0.00865,
    }
    synaptic_delays_prox = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    # all NMDA weights are zero; pass None explicitly
    net.add_evoked_drive(
        "evprox1",
        mu=26.61,
        sigma=2.47,
        numspikes=1,
        weights_ampa=weights_ampa_p1,
        weights_nmda=None,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
        event_seed=event_seed(),
    )

    # Second proximal evoked drive. NB: only AMPA weights differ from first
    weights_ampa_p2 = {
        "L2_basket": 0.000003,
        "L2_pyramidal": 1.438840,
        "L5_basket": 0.008958,
        "L5_pyramidal": 0.684013,
    }
    # all NMDA weights are zero; omit weights_nmda (defaults to None)
    net.add_evoked_drive(
        "evprox2",
        mu=137.12,
        sigma=8.33,
        numspikes=1,
        weights_ampa=weights_ampa_p2,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
        event_seed=event_seed(),
    )

    return net



def set_network_weights(params=None):

    """
    description: sets network to default values for an ERP as described in hnn tutorial
    """

    net = jones_2009_model()


    weights_ampa_p1 = {
        "L2_basket": params[0],
        "L2_pyramidal": params[1],
        "L5_basket": params[2],
        "L5_pyramidal": 0.00865,
    }

    weights_nmda_p1 = {
       "L2_basket": 0.08831,
        "L2_pyramidal": 0.01525,
        "L5_basket": params[3],
        "L5_pyramidal": params[4],
    }

    synaptic_delays_prox = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    # all NMDA weights are zero; pass None explicitly

    net.add_evoked_drive(
        "evprox1",
        mu=params[5],
        sigma=2.47,
        numspikes=1,
        weights_ampa=weights_ampa_p1,
        weights_nmda=weights_nmda_p1,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
        event_seed=event_seed(),
    )


    if (len(params)==6):

        return net

    weights_ampa_d1 = {
        "L2_basket": params[7],
        "L2_pyramidal": params[6],
        "L5_pyramidal": 0.142300,
        #"L5_basket": params[7],
    }
    weights_nmda_d1 = {
        "L2_basket": params[10],
        "L2_pyramidal": params[8],
        "L5_pyramidal": params[9],
        #"L5_basket": params[10],
    }
    synaptic_delays_d1 = {"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1}
    net.add_evoked_drive(
        "evdist1",
        mu=params[11],
        sigma=3.85,
        numspikes=1,
        weights_ampa=weights_ampa_d1,
        weights_nmda=weights_nmda_d1,
        location="distal",
        synaptic_delays=synaptic_delays_d1,
        event_seed=event_seed(),
    )

    if (len(params)==12):

        return net

    # Second proximal evoked drive. NB: only AMPA weights differ from first
    weights_ampa_p2 = {
        #"L2_basket": params[12],
        "L2_pyramidal": params[12],
        #"L5_basket": params[13],
        "L5_pyramidal": params[13],
    }

    weights_nmda_p2 = {
        #"L2_basket": params[12],
        "L2_pyramidal": params[14],
        #"L5_basket": params[16],
        "L5_pyramidal": params[15],
    }

    synaptic_delays_prox2 = {
        "L2_pyramidal": 0.1,
        #"L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    # all NMDA weights are zero; omit weights_nmda (defaults to None)

    net.add_evoked_drive(
        "evprox2",
        mu=params[16],
        sigma=8.33,
        numspikes=1,
        weights_ampa=weights_ampa_p2,
        weights_nmda = weights_nmda_p2,
        location="proximal",
        synaptic_delays=synaptic_delays_prox2,
        event_seed=event_seed(),
    )

    return net


def set_network_weights_small_steps(params=None):

    """
    description: sets network to default values for an ERP as described in hnn tutorial
    """

    net = jones_2009_model()


    weights_ampa_p1 = {
        "L2_basket": params[0],
    }

    weights_nmda_p1 = {
       "L2_basket": params[1],
    }

    if (len(params)==2):
        return net

    weights_ampa_p1 = {
        "L2_basket": params[0],
        "L2_pyramidal": params[2],
    }

    weights_nmda_p1 = {
       "L2_basket": params[1],
        "L2_pyramidal":params[3],
    }

    if (len(params)==4):
        return net

    weights_ampa_p1 = {
        "L2_basket": params[0],
        "L2_pyramidal": params[2],
        "L5_basket": params[4],
 
    }

    weights_nmda_p1 = {
       "L2_basket": params[1],
        "L2_pyramidal":params[3],
        "L5_basket": params[5],

    }

    if (len(params)==6):
        return net

    weights_ampa_p1 = {
        "L2_basket": params[0],
        "L2_pyramidal": params[2],
        "L5_basket": params[4],
        "L5_pyramidal": params[6],
    }

    weights_nmda_p1 = {
       "L2_basket": params[1],
        "L2_pyramidal":params[3],
        "L5_basket": params[5],
        "L5_pyramidal": params[7],
    }

    synaptic_delays_prox = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    if (len(params)==8):
        set_proximal1(net, weights_ampa_p1, weights_nmda_p1, synaptic_delays_prox)

        return net

    set_proximal1(net, weights_ampa_p1, weights_nmda_p1, synaptic_delays_prox, mu=params[8])

    if (len(params)==9):

        return net

    weights_ampa_d1 = {
        "L2_basket": params[9],
    }
    weights_nmda_d1 = {
        "L2_basket": params[10],

    }
    synaptic_delays_d1 = {
        "L2_basket": 0.1}

    set_distal(net, weights_ampa_d1, weights_nmda_d1, synaptic_delays_d1)

    
 
    if (len(params)==11):

        return net

    weights_ampa_d1 = {
        "L2_basket": params[9],
        "L2_pyramidal": params[11],

    }
    weights_nmda_d1 = {
        "L2_basket": params[10],
        "L2_pyramidal": params[12],

    }

    synaptic_delays_d1 = {
    "L2_basket": 0.1,
    "L2_pyramidal": 0.1}


    set_distal(net, weights_ampa_d1, weights_nmda_d1, synaptic_delays_d1)

    if (len(params)==13):

        return net
        
    weights_ampa_d1 = {
        "L2_basket": params[9],
        "L2_pyramidal": params[11],
        "L5_pyramidal": params[13],

    }
    weights_nmda_d1 = {
        "L2_basket": params[10],
        "L2_pyramidal": params[12],
        "L5_pyramidal": params[14],

    }

    synaptic_delays_d1 = {
    "L2_basket": 0.1,
    "L2_pyramidal": 0.1,
    "L5_pyramidal": 1.0}

    

    if (len(params)==15):

        set_distal(net, weights_ampa_d1, weights_nmda_d1, synaptic_delays_d1)

        return net


    set_distal(net, weights_ampa_d1, weights_nmda_d1, synaptic_delays_d1, mu = params[15])


    if (len(params)==16):

        return net

    # Second proximal evoked drive. NB: only AMPA weights differ from first
    weights_ampa_p2 = {
        "L2_basket": params[16],
    }

    weights_nmda_p2 = {
        "L2_basket": params[17],

    }


    synaptic_delays_p2 = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    if (len(params)==18):
        set_proximal2(net, weights_ampa_p2, weights_nmda_p2, synaptic_delays_p2)

        return net

    weights_ampa_p2 = {
        "L2_basket": params[16],
        "L2_pyramidal": params[18],
    }

    weights_nmda_p2 = {
        "L2_basket": params[17],
        "L2_pyramidal": params[19],

    }


    synaptic_delays_p2 = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
    }

    if (len(params)==20):
        set_proximal2(net, weights_ampa_p2, weights_nmda_p2, synaptic_delays_p2)

        return net

    weights_ampa_p2 = {
        "L2_basket": params[16],
        "L2_pyramidal": params[18],
        "L5_basket": params[20],

    }

    weights_nmda_p2 = {
        "L2_basket": params[17],
        "L2_pyramidal": params[19],
        "L5_basket": params[21],
    }


    synaptic_delays_p2 = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,

    }

    if (len(params)==22):
        set_proximal2(net, weights_ampa_p2, weights_nmda_p2, synaptic_delays_p2)

        return net

    weights_ampa_p2 = {
        "L2_basket": params[16],
        "L2_pyramidal": params[18],
        "L5_basket": params[20],
        "L5_pyramidal": params[22],
    }

    weights_nmda_p2 = {
        "L2_basket": params[17],
        "L2_pyramidal": params[19],
        "L5_basket": params[21],
        "L5_pyramidal": params[23],
    }


    synaptic_delays_p2 = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    if (len(params)==24):
        set_proximal2(net, weights_ampa_p2, weights_nmda_p2, synaptic_delays_p2)

        return net

    set_proximal2(net, weights_ampa_p2, weights_nmda_p2, synaptic_delays_p2, mu=params[24])


    return net


def set_network_weights_2_per_step(params=None):

    """
    description: sets network to default values for an ERP as described in hnn tutorial
    """

    net = jones_2009_model()


    weights_ampa_p1 = {
        "L2_basket": 0.08831,
        "L2_pyramidal": params[0],
        "L5_basket": 0.19934,
        "L5_pyramidal": 0.00865,
    }
    synaptic_delays_prox = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }


    # all NMDA weights are zero; pass None explicitly

    net.add_evoked_drive(
        "evprox1",
        mu=params[1],
        sigma=2.47,
        numspikes=1,
        weights_ampa=weights_ampa_p1,
        weights_nmda=None,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
        event_seed=event_seed(),
    )


    if (len(params)==2):
       
        return net


    weights_ampa_d1 = {
        "L2_basket": 0.006562,
        "L2_pyramidal": 0.000007,
        "L5_pyramidal": 0.142300,
    }
    weights_nmda_d1 = {
        "L2_basket": 0.019482,
        "L2_pyramidal": params[2],
        "L5_pyramidal": 0.080074,
    }


    synaptic_delays_d1 = {"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1}
    net.add_evoked_drive(
        "evdist1",
        mu=params[3],
        sigma=3.85,
        numspikes=1,
        weights_ampa=weights_ampa_d1,
        weights_nmda=weights_nmda_d1,
        location="distal",
        synaptic_delays=synaptic_delays_d1,
        event_seed=event_seed(),
    )

    if (len(params)==4):
        return net
    # Second proximal evoked drive. NB: only AMPA weights differ from first

    weights_ampa_p2 = {
        "L2_basket": 0.000003,
        "L2_pyramidal": 1.438840,
        "L5_basket": 0.008958,
        "L5_pyramidal": params[4],
    }

    # all NMDA weights are zero; omit weights_nmda (defaults to None)
    net.add_evoked_drive(
        "evprox2",
        mu=params[5],
        sigma=8.33,
        numspikes=1,
        weights_ampa=weights_ampa_p2,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
        event_seed=event_seed(),
    )

    return net



def set_network_1_params(params=None):

    """
    description: changes the network due to parameter settings drawn during sbi

    here one changes only the first distal drive and the second proximal drive, which was 
    used as toy example to see if the sbi is working for the case of the hnn simulator.
    """
    net = jones_2009_model()

    weights_ampa_p1 = {
        "L2_basket": 0.08831,
        "L2_pyramidal": 0.01525,
        "L5_basket": 0.19934,
        "L5_pyramidal": 0.00865,
    }
    synaptic_delays_prox = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    # all NMDA weights are zero; pass None explicitly
    net.add_evoked_drive(
        "evprox1",
        mu=params[0],
        sigma=2.47,
        numspikes=1,
        weights_ampa=weights_ampa_p1,
        weights_nmda=None,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
        event_seed=event_seed(),
    )
        # Second proximal evoked drive. NB: only AMPA weights differ from first
    weights_ampa_p2 = {
        "L2_basket": 0.000003,
        "L2_pyramidal": 1.438840,
        "L5_basket": 0.008958,
        "L5_pyramidal": 0.684013,
    }
    # all NMDA weights are zero; omit weights_nmda (defaults to None)
    net.add_evoked_drive(
        "evprox2",
        mu=137.12,
        sigma=8.33,
        numspikes=1,
        weights_ampa=weights_ampa_p2,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
        event_seed=event_seed(),
    )

    return net


def set_network_2_params(params=None):

    """
    description: changes the network due to parameter settings drawn during sbi

    here one changes only the first distal drive and the second proximal drive, which was 
    used as toy example to see if the sbi is working for the case of the hnn simulator.
    """

    net = jones_2009_model()
    weights_ampa_d1 = {
        "L2_basket": 0.006562,
        "L2_pyramidal": 0.000007,
        "L5_pyramidal": 0.142300,
    }
    weights_nmda_d1 = {
        "L2_basket": 0.019482,
        "L2_pyramidal": 0.004317,
        "L5_pyramidal": 0.080074,
    }
    synaptic_delays_d1 = {"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1}
    net.add_evoked_drive(
        "evdist1",
        mu=params[1],
        sigma=3.85,
        numspikes=1,
        weights_ampa=weights_ampa_d1,
        weights_nmda=weights_nmda_d1,
        location="distal",
        synaptic_delays=synaptic_delays_d1,
        event_seed=event_seed(),
    )

    weights_ampa_p1 = {
        "L2_basket": 0.08831,
        "L2_pyramidal": 0.01525,
        "L5_basket": 0.19934,
        "L5_pyramidal": 0.00865,
    }
    synaptic_delays_prox = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    # all NMDA weights are zero; pass None explicitly
    net.add_evoked_drive(
        "evprox1",
        mu=params[0],
        sigma=2.47,
        numspikes=1,
        weights_ampa=weights_ampa_p1,
        weights_nmda=None,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
        event_seed=event_seed(),
    )

    return net


def set_network_3_params(params=None):

    """
    description: changes the network due to parameter settings drawn during sbi

    here one changes only the first distal drive and the second proximal drive, which was 
    used as toy example to see if the sbi is working for the case of the hnn simulator.
    """

    net = jones_2009_model()
    weights_ampa_d1 = {
        "L2_basket": 0.006562,
        "L2_pyramidal": 0.000007,
        "L5_pyramidal": 0.142300,
    }
    weights_nmda_d1 = {
        "L2_basket": 0.019482,
        "L2_pyramidal": 0.004317,
        "L5_pyramidal": 0.080074,
    }
    synaptic_delays_d1 = {"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1}
    net.add_evoked_drive(
        "evdist1",
        mu=params[1],
        sigma=3.85,
        numspikes=1,
        weights_ampa=weights_ampa_d1,
        weights_nmda=weights_nmda_d1,
        location="distal",
        synaptic_delays=synaptic_delays_d1,
        event_seed=event_seed(),
    )

    weights_ampa_p1 = {
        "L2_basket": 0.08831,
        "L2_pyramidal": 0.01525,
        "L5_basket": 0.19934,
        "L5_pyramidal": 0.00865,
    }
    synaptic_delays_prox = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    # all NMDA weights are zero; pass None explicitly
    net.add_evoked_drive(
        "evprox1",
        mu=params[0],
        sigma=2.47,
        numspikes=1,
        weights_ampa=weights_ampa_p1,
        weights_nmda=None,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
        event_seed=event_seed(),
    )

    # Second proximal evoked drive. NB: only AMPA weights differ from first
    weights_ampa_p2 = {
        "L2_basket": 0.000003,
        "L2_pyramidal": 1.438840,
        "L5_basket": 0.008958,
        "L5_pyramidal": 0.684013,
    }
    # all NMDA weights are zero; omit weights_nmda (defaults to None)
    net.add_evoked_drive(
        "evprox2",
        mu=params[2],
        sigma=8.33,
        numspikes=1,
        weights_ampa=weights_ampa_p2,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
        event_seed=event_seed(),
    )

    return net


def set_proximal1(net, weights_ampa_p1, weights_nmda_p1, synaptic_delays_prox, mu=18.98):
    net.add_evoked_drive(
    "evprox1",
    mu=mu,
    sigma=3.68,
    numspikes=1,
    weights_ampa=weights_ampa_p1,
    weights_nmda=weights_nmda_p1,
    location="proximal",
    synaptic_delays=synaptic_delays_prox,
    event_seed=event_seed(),
    )

def set_distal(net, weights_ampa_d1, weights_nmda_d1, synaptic_delays_d1, mu=63.08):
    net.add_evoked_drive(
    "evdist1",
    mu=mu,
    sigma=3.85,
    numspikes=1,
    weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1,
    location="distal",
    synaptic_delays=synaptic_delays_d1,
    event_seed=event_seed(),
)

def set_proximal2(net, weights_ampa_p2, weights_nmda_p2, synaptic_delays_p2, mu=120):
    net.add_evoked_drive(
    "evprox2",
    mu=mu,
    sigma=10.3,
    numspikes=1,
    weights_ampa=weights_ampa_p2,
    weights_nmda = weights_nmda_p2,
    location="proximal",
    synaptic_delays=synaptic_delays_p2,
    event_seed=event_seed(),
)
