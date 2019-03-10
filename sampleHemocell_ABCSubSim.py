import os
import shutil

import numpy as np
from scipy.stats import norm,uniform,multivariate_normal

import pandas as pd

import UQLib.calibration.ABCSubSim as ABCSubSim
import hemocell.model as hemocell

from lxml import etree

#from local_config import *
from lisa_config import *

# Set seed for reproducibility
np.random.seed(6345789)

def distance(a,b):
    return np.sqrt(np.sum((a-b)**2))

def model_priors():
    kLink_prior = lambda sample: uniform.pdf(sample,15.0,185.0)
    kBend_prior = lambda sample: uniform.pdf(sample,80.0,120.0)
    viscosityRatio_prior = lambda sample: uniform.pdf(sample,5.0,10.0)
    return [kLink_prior,kBend_prior]

def model_samplers():
    kLink_sampler = lambda n_samples: np.random.uniform(15.0,200.0,n_samples)
    kBend_sampler = lambda n_samples: np.random.uniform(80.0,200.0,n_samples)
    viscosityRatio_sampler = lambda n_samples: np.random.uniform(5.0,15.0,n_samples)
    return [kLink_sampler,kBend_sampler]

if __name__ == "__main__":
    # Define problem parameters
    model_params = ["kLink","kBend"]
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

    # Construct problem dict
    problem = {"model_type":"external",
               "setup":(lambda params: hemocell.setup(modelpath,params)),
               "measure":(lambda outpath: hemocell.measureEI(8000,outpath)),
               "distance":distance,
               "model_params":model_params,
               "design_vars":design_vars,
               "input_data":design_vals,
               "output_data":el,
               "data_errors":el_err,
               "priors":model_priors(),
               "samplers":model_samplers()
              }
    
    # Sample from the posterior distribution
    os.makedirs("ABCSubSim_output")
    os.chdir("ABCSubSim_output")
    
    ABCSubSim_sampler = ABCSubSim.ABCSubSim(problem,logpath="%s/ABCSubSim_Hemocell_normal_3_12_log.pkl" % libpath, 
                                            logstep=10, tol=0.5, max_stages=5, invPa=5, nprocs=16)
    
    #ABCSubSim_sampler = ABCSubSim.load_state("%s/ABCSubSim_Hemocell_normal_3_12_log.pkl" % libpath)
    #ABCSubSim_sampler = ABCSubSim.ABCSubSim(problem,logpath=None,nprocs=16)

    df,qoi = ABCSubSim_sampler.sample(100,checkpoint=False)
    
    os.chdir("..")

    # Remove garbage
    #shutil.rmtree("./ABCSubSim_output")
    
    # Write output to files
    df.to_csv("ABCSubSim_hemocell_samples_normal_3_12.csv",sep=";",index=False)
    np.save("ABCSubSim_hemocell_qoi_normal_3_12.npy",qoi)
