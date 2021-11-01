import sys, os, datetime
import torch
import json
import datetime


class WriteToFile:
    """
    Class to store metadata and results in folder in order to work with them later or refer to e.g.
    parmeter settings
    
    stores the following:
    - prior interval
    - posterior samples
    - observations (true or simulated)
    - metadata like number of simulations, time, functions used, copy of python file for execution
    
    goal is to save everything into one folder
    """
    def __init__(
        self,
        path_parent: str = "results/",
        true_params: list = [] ,
        num_sim: int = None,
        experiment: str = 'erp'     
    ):
        self.date = datetime.datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
        self.path_parent = path_parent
        self.experiment = experiment
        self.num_sim = num_sim
        self.true_params = true_params



        self.folder = path_parent + self.experiment + self.date


        os.mkdir(self.folder)
        torch.save(true_params, '{}/true_params.pt'.format(self.folder))




    def create_file(self, file_name):
        os.chdir(self.folder)
        os.system('touch ' + file_name)
        os.chdir(os.pardir)
        os.chdir(os.pardir)
    

    def save_posterior(self, posterior):
    
        self.create_file('posterior.pt')
        torch.save(posterior, '{}/posterior.pt'.format(self.folder))
    

    def save_prior(self, prior):

        torch.save(prior, '{}/prior.pt'.format(self.folder))
        self.prior = prior

    def save_observations(self, x):
        torch.save(x, '{}/obs.pt'.format(self.folder))
    

    def save_thetas(self, thetas):
        torch.save(thetas, '{}/thetas.pt'.format(self.folder))
   

    def save_fig(self, fig):
        fig.savefig('{}/figure'.format(self.folder))

    
    def save_class(self):
        json_dict = {'date':self.date,
        'path':self.folder,
        'experiment name':self.experiment,
        'number of simulations':self.num_sim}
        with open(self.folder+'/meta.json', 'a') as f:
            json.dump(json_dict, f)
            f.close()

   



