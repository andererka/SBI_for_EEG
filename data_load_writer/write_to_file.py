import sys, os, datetime
import torch
import json
import datetime
import pickle


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

    

    def save_posterior(self, posterior):
        file_name = '{}/posterior.pt'.format(self.folder)
        if os.path.isfile(file_name):
            expand = 1
        while True:
            expand += 1
            new_file_name = file_name.split(".pt")[0] + str(expand) + ".pt"
            if os.path.isfile(new_file_name):
                continue
            else:
                file_name = new_file_name
                break
        torch.save(posterior, file_name)
    

    def save_prior(self, prior):

        torch.save(prior, '{}/prior.pt'.format(self.folder))
        self.prior = prior

    def save_observations(self, x):
        torch.save(x, '{}/obs.pt'.format(self.folder))
    

    def save_thetas(self, thetas):
        torch.save(thetas, '{}/thetas.pt'.format(self.folder))
   

    def save_fig(self, fig):
        file_name = '{}/figure.png'.format(self.folder)
        if os.path.isfile(file_name):
            expand = 1
        while True:
            expand += 1
            new_file_name = file_name.split(".png")[0] + str(expand) + ".png"
            if os.path.isfile(new_file_name):
                continue
            else:
                file_name = new_file_name
                break
        fig.savefig(file_name)

    
    def save_meta(self):
        json_dict = {'date':self.date,
        'path':self.folder,
        'experiment name':self.experiment,
        'number of simulations':self.num_sim}
        with open(self.folder+'/meta.json', 'a') as f:
            json.dump(json_dict, f)
            f.close()

    def save_all(self, posterior, prior, theta, x, fig):
        self.save_posterior(posterior)
        self.save_prior(prior)
        self.save_thetas(theta)
        self.save_observations(x)
        self.save_fig(fig)
        self.save_meta()



   



