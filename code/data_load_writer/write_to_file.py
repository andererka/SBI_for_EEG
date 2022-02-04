import sys, os, datetime
import torch
import json
import datetime
import pickle
import shutil


class WriteToFile(object):
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
        true_params: list = [],
        num_sim: int = None,
        experiment: str = "erp",
        density_estimator="maf",
        num_params=None,
        num_samples=None,
        slurm = True,
       
    ):
        self.date = datetime.datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
        self.path_parent = path_parent
        self.experiment = experiment
        self.num_sim = num_sim
        self.true_params = true_params
        self.density_estimator = density_estimator
        self.num_params = num_params
        self.num_samples = num_samples

        save_to = '/mnt/qb/work/macke/kanderer29/'

        if (slurm== False):
            self.folder = path_parent + self.experiment 
            
        else:
            self.folder = save_to + path_parent + self.experiment 



    def save_posterior(self, posterior):

        file_name = "posterior.pt"
        file_name = make_file_name(file_name)
        torch.save(posterior, file_name)

    def save_prior(self, prior):


        torch.save(prior, "prior.pt")


    def save_proposal(self, prop, name='default'):

        file_name = "proposal.pt"
        file_name = make_file_name(file_name)

        if (name=='default'):
            torch.save(prop, file_name)
        else:
            torch.save(prop, "{}/{}".format(name, file_name))

    def save_observations(self, x, name='default'):

        file_name = "obs.pt"
        file_name = make_file_name(file_name)

        if (name=='default'):
            torch.save(x, file_name)
        else:
            torch.save(x, "{}/{}".format(name, file_name))

    def save_obs_without(self, x_without, name='default'):

        file_name = "obs_without.pt"
        file_name = make_file_name(file_name)
        print(file_name)

        if (name=='default'):
            torch.save(x_without, file_name)
        else:
            torch.save(x_without, "{}/{}".format(name, file_name))

    def save_thetas(self, thetas, name='default'):

        file_name = "thetas.pt"
        file_name = make_file_name(file_name)
        print(file_name)

        if (name=='default'):
            torch.save(thetas, file_name)
        else:
            torch.save(thetas, "{}/{}".format(name, file_name))

    def save_fig(self, fig, figname=None):

        file_name = figname     
        if file_name== None:
            file_name = "{}/fig.png".format(self.folder)


        else:
            file_name = make_file_name(file_name) 
            file_name = "{}/figure{}.png".format(self.folder, file_name)

        fig.savefig(file_name)

    def save_meta(self, start_time, finish_time):

        json_dict = {
            "date": self.date,
            "path": self.folder,
            "experiment name": self.experiment,
            "number of simulations": self.num_sim,
            "number of samples": self.num_samples,
            "number of parameters": self.num_params,
            "type of density estimator": self.density_estimator,
            "start time:": start_time,
            "finish_time": finish_time,
        }

        file_name = make_file_name('meta') 

        with open(self.folder + "/" + file_name, "a") as f:
            json.dump(json_dict, f)
            f.close()

    def save_all(
        self,
        start_time=None,
        finish_time=None,
        source=__file__,
    ):


        self.save_meta(start_time, finish_time)

        source = "{}.py".format(source)
        # Destination path
        destination = self.folder

        try:
            shutil.copyfile(source, destination)
            print("File copied successfully.")
        except:
            print("File could not be copied")


def make_file_name(file_name):
    if os.path.isfile(file_name):
        expand = 1
        while True:
            expand += 1
            new_file_name = file_name + str(expand) + ".pt"
            if os.path.isfile(new_file_name):
                continue
            else:
                file_name = new_file_name
                break
