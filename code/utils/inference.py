
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi

def run_sim_inference(prior, simulation_wrapper, num_simulations=1000, density_estimator='nsf'):


    #posterior = infer(simulation_wrapper, prior, method='SNPE_C', 
                  #num_simulations=number_simulations, num_workers=4)     

    simulator, prior = prepare_for_sbi(simulation_wrapper, prior)
    inference = SNPE(prior, density_estimator=density_estimator)


    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_simulations)

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