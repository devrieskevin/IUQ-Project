import os
import shutil

import numpy as np
from scipy.stats import norm,uniform,multivariate_normal

import pandas as pd

import UQLib.calibration.TMCMC as TMCMC
import hemocell.model as hemocell

from lxml import etree

#from local_config import *
from lisa_config import *

# Set seed for reproducibility
np.random.seed(6345789)

def model_prior(sample):
    kLink_prior = uniform.pdf(sample[0],15.0,185.0)
    kBend_prior = uniform.pdf(sample[1],80.0,120.0)
    #viscosityRatio_prior = uniform.pdf(sample[2],5.0,10.0)
    return np.prod([kLink_prior,kBend_prior])

def model_sampler(n_samples):
    kLink_samples = np.random.uniform(15.0,200.0,n_samples)
    kBend_samples = np.random.uniform(80.0,200.0,n_samples)
    #viscosityRatio_samples = np.random.uniform(5.0,15.0,n_samples)
    return np.column_stack([kLink_samples,kBend_samples])

def error_prior(sample):
    return uniform.pdf(sample[0],0.001,0.999)

def error_sampler(n_samples):
    return np.random.uniform(0.001,1.0,(n_samples,1))

if __name__ == "__main__":
    # Define problem parameters
    model_params = ["kLink","kBend"]
    error_params = ["model_uncertainty"]
    design_vars = ["shearrate"]
    
    # Extract data from dataset
    data = pd.read_csv("%s/Ekcta_100.csv" % (datapath),sep=";")
    data = data.loc[data["Treatment"] == 0.5]
    stress,el,el_err = data.values[3:12,[1,3,4]].T

    # Get data from config files
    configpath = "%s/hemocell/templates/config_template.xml" % (libpath)
    tree = etree.parse(configpath, parser=etree.XMLParser(remove_blank_text=True, remove_comments=True))
    root = tree.getroot()
    nuP = float(root.find("domain/nuP").text)
    rhoP = float(root.find("domain/rhoP").text)

    # Compute the shear rates
    shearrate = stress / (nuP * rhoP)
    design_vals = shearrate

    # Map model errors to data samples
    error_mapping = [error_params[0] for n in range(shearrate.shape[0])]

    # Construct problem dict
    problem = {"model_type":"external",
               "setup":(lambda params: hemocell.setup(modelpath,params)),
               "measure":(lambda outpath: hemocell.measureEI(8000,outpath)),
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
    os.makedirs("TMCMC_output")
    os.chdir("TMCMC_output")
    
    TMCMC_sampler = TMCMC.TMCMC(problem,logpath="%s/TMCMC_Hemocell_normal_3_12_log.pkl" % libpath,nprocs=16)
    #TMCMC_sampler = TMCMC.load_state("%s/TMCMC_Hemocell_normal_3_12_log.pkl" % libpath)
    #TMCMC_sampler = TMCMC.TMCMC(problem,logpath=None,nprocs=16)

    df,qoi = TMCMC_sampler.sample(1000,checkpoint=False)
    
    os.chdir("..")

    # Remove garbage
    #shutil.rmtree("./TMCMC_output")
    
    # Write output to files
    df.to_csv("hemocell_samples_normal_3_12.csv",sep=";",index=False)
    np.save("hemocell_qoi_normal_3_12.npy",qoi)
