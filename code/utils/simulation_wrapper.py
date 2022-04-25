
from hnn_core import simulate_dipole, jones_2009_model
from pyro import param

import torch
import numpy as np


from random import randrange


def event_seed():
    """
    description: makes sure that one does not take the same random seed for each simulation as it would be the default in the hnn core code;
    permalink to the hnn code location: https://github.com/jonescompneurolab/hnn-core/blob/0406ed1a2b2335b786e83eb1698f27a5c3dcdadc/hnn_core/drives.py#L262
    """
    seed = randrange(2000)
    return seed


class SimulationWrapper:
    """
    Class in order to select and call a simulation wrapper according to:
    - num_params: number of parameters that are to be inferred in total
    - noise: noise can be added to observation to induce more stochasticity. Default is set to True
    
    """

    def __init__(self, num_params = 17,  noise = True):
        self.num_params = num_params
        self.noise = noise

    

    def __call__(self, params):
        if (self.num_params == 17 or self.num_params == 20):
            return self.simulation_wrapper_all(params)

            
        elif (self.num_params == 25):
            return self.simulation_wrapper_25(params)



    def simulation_wrapper_all(self, params):  # input possibly array of 1 or more params
        """
        simulation wrapper for up to 17 params
        -sets weights for hnn simulator according to the drawn parameters

        simulation stops earlier (after ~70ms) if only the weights for the first proximal drive 
        are changed and stops also earlier if the weights for the first proximal drive + the weights
        for the distal drive are changed, but not the ones for the second proximal drive (~120ms).

        Output [torch.tensor]: observation of simulated ERP
        """

        early_stop = 200.0

        # get number of parameters:
        if len(params.size()) == 2:
            param_size = int(params.size()[1])
        elif len(params.size()) == 1:
            param_size = int(params.size()[0])

        print('param size', param_size)

        if (param_size == 6 and self.num_params == 17):
            
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

        if param_size==17:
    
            net = set_network_weights(params)
        if param_size == 20:
            net = set_network_weights_std(params)


        window_len, scaling_factor = 30, 3000

        dpls = simulate_dipole(net, tstop=early_stop, n_trials=1)
        for dpl in dpls:


            obs = dpl.smooth(window_len).scale(scaling_factor).data["agg"]

            # make time series more stochastic:
            if self.noise == True:
                noise = np.random.normal(0, 1, obs.shape[0])

                obs += noise


            print('obs', obs)


        return torch.from_numpy(obs)


    def simulation_wrapper_25(self, params):  # input possibly array of 1 or more params
        """
        simulation wrapper for 25 parameters

        simulation stops earlier (after ~70ms) if only the weights for the first proximal drive 
        are changed and stops also earlier if the weights for the first proximal drive + the weights
        for the distal drive are changed, but not the ones for the second proximal drive (~120ms).

        Output [torch.tensor]: observation of simulated ERP
        """

        early_stop = 200.0
  
  
        if len(params.size()) == 2:
          
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

    
        net = set_network_weights_steps_for_25(params)



        window_len, scaling_factor = 30, 3000

        dpls = simulate_dipole(net, tstop=early_stop, n_trials=1)
        for dpl in dpls:
            obs = dpl.smooth(window_len).scale(scaling_factor).data["agg"]
            
            # make time series more stochastic:
            if self.noise == True:
                noise = np.random.normal(0, 1, obs.shape[0])

                obs += noise

        return torch.from_numpy(obs)




def set_network_weights_steps_for_25(params=None):

    """
    description: sets parameters for proximal and distal drive. more details can be found in the hnn tutorial https://jonescompneurolab.github.io/hnn-tutorials/
    """

    net = jones_2009_model()

    if any(isinstance(el, list) for el in params):
        num_params = len(params[0])
    else:
        num_params = len(params)
 



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


    set_proximal1(net, weights_ampa_p1, weights_nmda_p1, synaptic_delays_prox, mu=params[8])

    if (num_params==9):

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



    set_distal(net, weights_ampa_d1, weights_nmda_d1, synaptic_delays_d1, mu = params[15])


    if (num_params==16):

        return net

    # Second proximal evoked drive. NB: only AMPA weights differ from first


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

    set_proximal2(net, weights_ampa_p2, weights_nmda_p2, synaptic_delays_p2, mu=params[24])

    if (num_params == 25):
        return net
    else:
        print('number of parameters not implemneted')
        exit()




def set_proximal1(net, weights_ampa_p1, weights_nmda_p1, synaptic_delays_prox, mu=18.98):
    net.add_evoked_drive(
    "evprox1",
    mu=mu,
    sigma=0.01,
    numspikes=1,
    event_seed = event_seed(),
    weights_ampa=weights_ampa_p1,
    weights_nmda=weights_nmda_p1,
    location="proximal",
    synaptic_delays=synaptic_delays_prox,
    )

def set_distal(net, weights_ampa_d1, weights_nmda_d1, synaptic_delays_d1, mu=63.08):
    net.add_evoked_drive(
    "evdist1",
    mu=mu,
    sigma=0.01,
    numspikes=1,
    event_seed = event_seed(),
    weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1,
    location="distal",
    synaptic_delays=synaptic_delays_d1,
)

def set_proximal2(net, weights_ampa_p2, weights_nmda_p2, synaptic_delays_p2, mu=120):
    net.add_evoked_drive(
    "evprox2",
    mu=mu,
    sigma=0.01,
    numspikes=1,
    event_seed = event_seed(),
    weights_ampa=weights_ampa_p2,
    weights_nmda = weights_nmda_p2,
    location="proximal",
    synaptic_delays=synaptic_delays_p2,
)


def set_network_weights(params=None):

    """
    description: sets network for the drives for an ERP as described in hnn tutorial https://jonescompneurolab.github.io/hnn-tutorials/
    """

    net = jones_2009_model()

    if any(isinstance(el, list) for el in params):
        num_params = len(params[0])
    else:
        num_params = len(params)

    print('num_params', num_params)


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
        #sigma=0.01,
        sigma = 0,
        numspikes=1,
        event_seed = event_seed(),
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
        #sigma=0.01,
        sigma = 0,
        event_seed = event_seed(),
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
        #sigma=0.01,
        sigma = 0,
        numspikes=1,
        weights_ampa=weights_ampa_p2,
        weights_nmda = weights_nmda_p2,
        event_seed = event_seed(),
        location="proximal",
        synaptic_delays=synaptic_delays_prox2,
    )

    return net



def set_network_weights_std(params=None):

    """
    description: sets network for the drives for an ERP as described in hnn tutorial https://jonescompneurolab.github.io/hnn-tutorials/
    """

    net = jones_2009_model()

    if any(isinstance(el, list) for el in params):
        num_params = len(params[0])
    else:
        num_params = len(params)

    print('num_params', num_params)


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
        #sigma=0.01,
        #sigma = 0,
        sigma = params[6],
        numspikes=1,
        event_seed = event_seed(),
        weights_ampa=weights_ampa_p1,
        weights_nmda=weights_nmda_p1,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
    )


    if (num_params==6):

        return net

    weights_ampa_d1 = {
        "L2_basket": params[8],
        "L2_pyramidal": params[7],
        "L5_pyramidal": 0.142300,
        #"L5_basket": params[7],
    }
    weights_nmda_d1 = {
        "L2_basket": params[11],
        "L2_pyramidal": params[9],
        "L5_pyramidal": params[10],
        #"L5_basket": params[10],
    }
    synaptic_delays_d1 = {"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1}
    net.add_evoked_drive(
        "evdist1",
        mu=params[12],
        #sigma=0.01,
        #sigma = 0,
        sigma = params[13],
        event_seed = event_seed(),
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
        "L2_pyramidal": params[14],
        #"L5_basket": params[13],
        "L5_pyramidal": params[15],
    }

    weights_nmda_p2 = {
        #"L2_basket": params[12],
        "L2_pyramidal": params[16],
        #"L5_basket": params[16],
        "L5_pyramidal": params[17],
    }

    synaptic_delays_prox2 = {
        "L2_pyramidal": 0.1,
        #"L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    # all NMDA weights are zero; omit weights_nmda (defaults to None)

    net.add_evoked_drive(
        "evprox2",
        mu=params[18],
        #sigma=0.01,
        #sigma = 0,
        sigma = params[19],
        numspikes=1,
        weights_ampa=weights_ampa_p2,
        weights_nmda = weights_nmda_p2,
        event_seed = event_seed(),
        location="proximal",
        synaptic_delays=synaptic_delays_prox2,
    )

    return net