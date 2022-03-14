import hnn_core
from hnn_core import simulate_dipole, jones_2009_model
from summary_features.calculate_summary_features import (
    calculate_summary_stats_number
)
import torch

import numpy as np


class SimulationWrapper:
    """
    Class in order to select and call a simulation wrapper according to:
    - num_params: number of parameters that are to be inferred in total
    - change_order: default False; if True, calls a simulation wrapper that takes another 
                    inference order. Could test if the inference order plays a role or if
                    inference is robust
    - small_steps: if True, takes incremental steps of 2. otherwise parameter sets consist of
                    about 6 parameters
    
    """

    def __init__(self, num_params = 17, change_order = False):
        self.num_params = num_params

        # not implemented so far in simulation function
        self.change_order = change_order
    

    def __call__(self, params):
        if (self.num_params == 17 or self.num_params == 6):
            if (self.change_order == False):
                return self.simulation_wrapper_all(params)
            
        elif (self.num_params == 25):
            return self.simulation_wrapper_all_small_steps(params)



    def simulation_wrapper_all(self, params):  # input possibly array of 1 or more params
        """
        simulation wrapper for the neural incremental approach where wrapper can take 
        different number of parameters and then sets weights for hnn simulator
        according to the drawn parameters

        simulation stops earlier (after ~70ms) if only the weights for the first proximal drive 
        are changed and stops also earlier if distal weights are changed (~120ms)

        Output [torch.tensor]: observation of simulated ERP
        """

        early_stop = 200.0

        if len(params.size()) == 2:
            print('dim=1')
            param_size = int(params.size()[1])
        elif len(params.size()) == 1:
            param_size = int(params.size()[0])

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

            # make time series more stochastic:
            noise = np.random.normal(0, 0.1, obs.shape[0])

            obs += noise


            print('obs', obs)


        return torch.from_numpy(obs)


    def simulation_wrapper_all_small_steps(self, params):  # input possibly array of 1 or more params
        """
        Input: Parameter list; can be of different size: 2, 4, 6, 8, 9, 11, 13, 15, 16, 18, 20, 22, 24 or 25

        simulation wrapper where wrapper can take 
        different number of parameters and then sets weights for hnn simulator
        according to the drawn parameters

        simulation stops earlier (after ~70ms) if number of parameters smaller
        than 10 and stops also earlier if number of parameters smaller than 17 (~120ms)

        Output [torch.tensor]: observation of simulated ERP
        
        """

        early_stop = 200.0
  
        print(params.size())
        if len(params.size()) == 2:
            print('dim=1')
            param_size = int(params.size()[1])
        elif len(params.size()) == 1:
            param_size = int(params.size()[0])

        if (param_size < 10):

            
            early_stop = 70.0

        elif (param_size >= 10 and param_size < 17):

            early_stop = 120.0

        
        print('early stop', early_stop)
        print('param size ', param_size)


        params = params.tolist()

        # add more stochasticity to params:
        #noise = np.random.normal(0, 0.1, param_size)

        #params += noise

        #print(params)

    
        if self.change_order == False:
            net = set_network_weights_small_steps(params)
        else:
            net = set_weights_small_steps_changed_order(params)


        window_len, scaling_factor = 30, 3000

        dpls = simulate_dipole(net, tstop=early_stop, n_trials=1)
        for dpl in dpls:
            obs = dpl.smooth(window_len).scale(scaling_factor).data["agg"]
            
            # make time series more stochastic:
            noise = np.random.normal(0, 0.5, obs.shape[0])

            obs += noise

        return torch.from_numpy(obs)





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
    )

    return net





def set_network_weights_small_steps(params=None):

    """
    description: sets network to default values for an ERP as described in hnn tutorial
    """

    net = jones_2009_model()

    if any(isinstance(el, list) for el in params):
        num_params = len(params[0])
    else:
        num_params = len(params)
    #num_params = num

    print('set network params:', params)


    weights_ampa_p1 = {
        "L2_basket": params[0],
    }

    weights_nmda_p1 = {
       "L2_basket": params[1],
    }

    synaptic_delays_prox = {
        "L2_basket": 0.1,
    }

    if (num_params==2):
        set_proximal1(net, weights_ampa_p1, weights_nmda_p1, synaptic_delays_prox)
        return net

    weights_ampa_p1 = {
        "L2_basket": params[0],
        "L2_pyramidal": params[2],
    }

    weights_nmda_p1 = {
       "L2_basket": params[1],
        "L2_pyramidal":params[3],
    }
    synaptic_delays_prox = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
    }

    if (num_params==4):
        set_proximal1(net, weights_ampa_p1, weights_nmda_p1, synaptic_delays_prox)
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
    synaptic_delays_prox = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
    }

    if (num_params==6):
        set_proximal1(net, weights_ampa_p1, weights_nmda_p1, synaptic_delays_prox)
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

    if (num_params==8):
        set_proximal1(net, weights_ampa_p1, weights_nmda_p1, synaptic_delays_prox)

        return net

    set_proximal1(net, weights_ampa_p1, weights_nmda_p1, synaptic_delays_prox, mu=params[8])

    if (num_params==9):

        return net

    weights_ampa_d1 = {
        "L2_basket": params[9],
    }
    weights_nmda_d1 = {
        "L2_basket": params[10],

    }
    synaptic_delays_d1 = {
        "L2_basket": 0.1}
    
 
    if (num_params==11):
        set_distal(net, weights_ampa_d1, weights_nmda_d1, synaptic_delays_d1)
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

    if (num_params==13):
        set_distal(net, weights_ampa_d1, weights_nmda_d1, synaptic_delays_d1)
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

    

    if (num_params==15):

        set_distal(net, weights_ampa_d1, weights_nmda_d1, synaptic_delays_d1)

        return net


    set_distal(net, weights_ampa_d1, weights_nmda_d1, synaptic_delays_d1, mu = params[15])


    if (num_params==16):

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
    }

    if (num_params==18):
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

    if (num_params==20):
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

    if (num_params==22):
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

    if (num_params==24):
        set_proximal2(net, weights_ampa_p2, weights_nmda_p2, synaptic_delays_p2)

        return net

    set_proximal2(net, weights_ampa_p2, weights_nmda_p2, synaptic_delays_p2, mu=params[24])

    if (num_params == 25):
        return net
    else:
        print('number of parameters not implemneted')
        exit()

def set_weights_small_steps_changed_order(params=None, num=2):

    """
    description: sets network to default values for an ERP as described in hnn tutorial
    """

    net = jones_2009_model()

    num_params = num

    weights_ampa_p1 = {

        "L5_pyramidal": params[0],
    }

    weights_nmda_p1 = {

        "L5_pyramidal": params[1],
    }

    synaptic_delays_prox = {

        "L5_pyramidal": 1.0,
    }


    if (num_params==2):
        set_proximal1(net, weights_ampa_p1, weights_nmda_p1, synaptic_delays_prox)
        return net

    weights_ampa_p1 = {

        "L5_basket": params[2],
        "L5_pyramidal": params[0],
    }

    weights_nmda_p1 = {

        "L5_basket": params[3],
        "L5_pyramidal": params[1],
    }

    synaptic_delays_prox = {
        "L5_basket": 0.1,
        "L5_pyramidal": 0.1,

    }

    if (num_params==4):
        set_proximal1(net, weights_ampa_p1, weights_nmda_p1, synaptic_delays_prox)
        return net

    weights_ampa_p1 = {

        "L2_pyramidal": params[4],
        "L5_basket": params[2],
        "L5_pyramidal": params[0],
    }

    weights_nmda_p1 = {

        "L2_pyramidal":params[5],
        "L5_basket": params[3],
        "L5_pyramidal": params[1],
    }

    synaptic_delays_prox = {

        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    if (num_params==6):
        set_proximal1(net, weights_ampa_p1, weights_nmda_p1, synaptic_delays_prox)
        return net

    weights_ampa_p1 = {
        "L2_basket": params[6],
        "L2_pyramidal": params[4],
        "L5_basket": params[2],
        "L5_pyramidal": params[0],
    }

    weights_nmda_p1 = {
       "L2_basket": params[7],
        "L2_pyramidal":params[5],
        "L5_basket": params[3],
        "L5_pyramidal": params[1],
    }

    synaptic_delays_prox = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    if (num_params==8):
        set_proximal1(net, weights_ampa_p1, weights_nmda_p1, synaptic_delays_prox)

        return net

    print(params)

    set_proximal1(net, weights_ampa_p1, weights_nmda_p1, synaptic_delays_prox, mu=params[8])

    if (num_params ==9):

        return net

    weights_ampa_d1 = {

        "L5_pyramidal": params[9],

    }
    weights_nmda_d1 = {

        "L5_pyramidal": params[10],

    }

    synaptic_delays_d1 = {

    "L5_pyramidal": 1.0}

    set_distal(net, weights_ampa_d1, weights_nmda_d1, synaptic_delays_d1)

    
 
    if (num_params ==11):

        return net

    weights_ampa_d1 = {

        "L2_pyramidal": params[11],
        "L5_pyramidal": params[9],

    }
    weights_nmda_d1 = {

        "L2_pyramidal": params[12],
        "L5_pyramidal": params[10],

    }

    synaptic_delays_d1 = {
        "L2_pyramidal": 0.1,
        "L5_pyramidal": 1.0}


    set_distal(net, weights_ampa_d1, weights_nmda_d1, synaptic_delays_d1)

    if (num_params ==13):

        return net
        
    weights_ampa_d1 = {
        "L2_basket": params[13],
        "L2_pyramidal": params[11],
        "L5_pyramidal": params[9],

    }
    weights_nmda_d1 = {
        "L2_basket": params[14],
        "L2_pyramidal": params[12],
        "L5_pyramidal": params[10],

    }

    synaptic_delays_d1 = {
    "L2_basket": 0.1,
    "L2_pyramidal": 0.1,
    "L5_pyramidal": 1.0}

    

    if (num_params ==15):

        set_distal(net, weights_ampa_d1, weights_nmda_d1, synaptic_delays_d1)

        return net


    set_distal(net, weights_ampa_d1, weights_nmda_d1, synaptic_delays_d1, mu = params[15])


    if (num_params == 16):

        return net

    weights_ampa_p2 = {

        "L5_pyramidal": params[16],
    }

    weights_nmda_p2 = {
 
        "L5_pyramidal": params[17],
    }


    synaptic_delays_p2 = {

        "L5_pyramidal": 1.0,
    }


    if (num_params ==18):
        set_proximal2(net, weights_ampa_p2, weights_nmda_p2, synaptic_delays_p2)

        return net

    weights_ampa_p2 = {

        "L5_basket": params[18],
        "L5_pyramidal": params[16],
    }

    weights_nmda_p2 = {

        "L5_basket": params[19],
        "L5_pyramidal": params[17],
    }


    synaptic_delays_p2 = {

        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }
    if (num_params ==20):
        set_proximal2(net, weights_ampa_p2, weights_nmda_p2, synaptic_delays_p2)

        return net

    weights_ampa_p2 = {

        "L2_pyramidal": params[20],
        "L5_basket": params[18],
        "L5_pyramidal": params[16],
    }

    weights_nmda_p2 = {

        "L2_pyramidal": params[21],
        "L5_basket": params[19],
        "L5_pyramidal": params[17],
    }


    synaptic_delays_p2 = {

        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    if (num_params ==22):
        set_proximal2(net, weights_ampa_p2, weights_nmda_p2, synaptic_delays_p2)

        return net

    weights_ampa_p2 = {
        "L2_basket": params[22],
        "L2_pyramidal": params[20],
        "L5_basket": params[18],
        "L5_pyramidal": params[16],
    }

    weights_nmda_p2 = {
        "L2_basket": params[23],
        "L2_pyramidal": params[21],
        "L5_basket": params[19],
        "L5_pyramidal": params[17],
    }


    synaptic_delays_p2 = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    if (num_params == 24):
        set_proximal2(net, weights_ampa_p2, weights_nmda_p2, synaptic_delays_p2)

        return net

    set_proximal2(net, weights_ampa_p2, weights_nmda_p2, synaptic_delays_p2, mu=params[24])


    return net


def set_network_weights_2_per_step(params=None):

    """
    description: sets network to default values for an ERP as described in hnn tutorial
    """

    net = jones_2009_model()

    if any(isinstance(el, list) for el in params):
        num_params = len(params[0])
    else:
        num_params = len(params)

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
    )


    if (num_params ==2):
       
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
    )

    if (num_params ==4):
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
)


def set_network_weights(params=None):

    """
    description: sets network to default values for an ERP as described in hnn tutorial
    """

    net = jones_2009_model()

    if any(isinstance(el, list) for el in params):
        num_params = len(params[0])
    else:
        num_params = len(params)


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
    )


    if (num_params==6):

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
    )

    if (num_params==12):

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
    )

    return net
