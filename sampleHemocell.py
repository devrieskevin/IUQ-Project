import os
import shutil

import numpy as np
from scipy.stats import norm,uniform,multivariate_normal

import pandas as pd

import UQLib.calibration.TMCMC as TMCMC
import hemocell.model as hemocell

import matplotlib.pyplot as plt

from lxml import etree

from local_config import *
#from lisa_config import *

np.random.seed(696969)


def model_prior(sample):
    pass

def model_sampler(n_samples):
    pass

def error_prior(sample):
    pass

def error_sampler(n_samples):
    pass

if __name__ == "__main__":
    
    # Define problem parameters
    model_params = ["kLink","kBend","viscosityRatio"]
    error_params = ["model_uncertainty"]
    design_vars = ["shearrate"]
    
    # Extract data from dataset
    data = pd.read_csv("%s/Ekcta_100.csv" % (datapath),sep=";")
    data = data.loc[data["Treatment"] == 0.5]
    stress,el,el_err = data.values[:,[1,3,4]].T

    # Get data from config files
    configpath = "%s/hemocell/templates/config_template.xml" % (libpath)
    tree = etree.parse(configpath, parser=etree.XMLParser(remove_blank_text=True, remove_comments=True))
    root = tree.getroot()
    nuP = float(root.find("domain/nuP").text)
    rhoP = float(root.find("domain/rhoP").text)

    # Compute the shear rate
    shearrate = stress / (nuP * rhoP)
    design_vals = shearrate

    # Map model errors to data samples
    error_mapping = [error_params[0] for n in range(shearrate.shape[0])]

    # Construct problem dict
    problem = {"model_type":"external",
               "setup":(lambda params: hemocell.setup(modelpath,params)),
               "measure":(lambda outpath: hemocell.measureEI(70000,outpath)),
               "model_params":model_params,
               "error_params":error_params,
               "design_vars":design_vars,
               "input_data":design_vals,
               "output_data":el,
               "data_errors":el_err,
               "error_mapping":error_mapping,
               "model_prior":model_prior,
               "model_sampler":model_sampler,
               "error_prior":error_prior,
               "error_sampler":error_sampler
              }
    
    # Sample from the posterior distribution
    #os.makedirs("TMCMC_output")
    #os.chdir("TMCMC_output")
    df,qoi = TMCMC.sample(problem,2,nprocs=16)
    #os.chdir("..")

    # Remove garbage
    #shutil.rmtree("./TMCMC_output")
    
    # Write output to 
    
    
    
    
    
    
    
