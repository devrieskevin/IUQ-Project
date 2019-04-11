import os
import shutil

import argparse

import numpy as np
from scipy.stats import norm,uniform,multivariate_normal

import pandas as pd

import UQLib.calibration.ABCSubSim as ABCSubSim
import hemocell.model as hemocell

from lxml import etree

#from local_config import *
from lisa_config import *

# Set seed for reproducibility
np.random.seed(77777)

def distance(a,b):
    return np.sqrt(np.mean((a-b)**2))

def model_priors(enableInteriorViscosity):
    kLink_prior = lambda sample: uniform.pdf(sample,15.0,285.0)
    kBend_prior = lambda sample: uniform.pdf(sample,80.0,220.0)
    viscosityRatio_prior = lambda sample: uniform.pdf(sample,1.0,49.0)
    
    if enableInteriorViscosity:
        return [kLink_prior,kBend_prior,viscosityRatio_prior]
    else:
        return [kLink_prior,kBend_prior]

def model_samplers(enableInteriorViscosity):
    kLink_sampler = lambda n_samples: np.random.uniform(15.0,300.0,n_samples)
    kBend_sampler = lambda n_samples: np.random.uniform(80.0,300.0,n_samples)
    viscosityRatio_sampler = lambda n_samples: np.random.uniform(1.0,50.0,n_samples)
    
    if enableInteriorViscosity:
        return [kLink_sampler,kBend_sampler,viscosityRatio_sampler]
    else:
        return [kLink_sampler,kBend_sampler]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--n_samples",dest="n_samples",type=int,default=100)
    parser.add_argument("--tmax",type=int,required=True)
    parser.add_argument("--enableInteriorViscosity",type=int,default=0)
    parser.add_argument("--checkpointed",dest="checkpointed",action="store_true",default=False)
    parser.add_argument("--imin",dest="imin",type=int,default=3)
    parser.add_argument("--imax",dest="imax",type=int,default=12)
    parser.add_argument("--nprocs",dest="nprocs",type=int,default=16)
    parser.add_argument("--model_type",dest="model_type",type=str,default="external")

    args = parser.parse_args()

    # Set design variable argument values
    n_samples = args.n_samples
    tmax = args.tmax
    enableInteriorViscosity = args.enableInteriorViscosity
    checkpointed = args.checkpointed
    imin = args.imin
    imax = args.imax
    nprocs = args.nprocs
    model_type = args.model_type

    # Define problem parameters
    if enableInteriorViscosity:
        model_params = ["kLink","kBend","viscosityRatio"]
    else:
        model_params = ["kLink","kBend"]

    design_vars = ["shearrate","tmax","tmeas","enableInteriorViscosity"]
    
    # Extract data from dataset
    data = pd.read_csv("%s/Ekcta_100.csv" % (datapath),sep=";")
    data = data.loc[data["Treatment"] == 0.5]
    stress,el,el_err = data.values[imin:imax,[1,3,4]].T

    # Get data from config files
    configpath = "%s/hemocell/templates/config_template.xml" % (libpath)
    tree = etree.parse(configpath, parser=etree.XMLParser(remove_blank_text=True, remove_comments=True))
    root = tree.getroot()
    nuP = float(root.find("domain/nuP").text)
    rhoP = float(root.find("domain/rhoP").text)

    # Compute the shear rates
    shearrate = stress / (nuP * rhoP)

    design_vals = np.row_stack(np.broadcast(shearrate,tmax,tmax,enableInteriorViscosity))

    # Construct problem dict
    problem = {"model_type":model_type,
               "setup":(lambda params: hemocell.setup(modelpath,params)),
               "measure":hemocell.measureEI,
               "distance":distance,
               "model_params":model_params,
               "design_vars":design_vars,
               "input_data":design_vals,
               "output_data":el,
               "data_errors":el_err,
               "priors":model_priors(enableInteriorViscosity),
               "samplers":model_samplers(enableInteriorViscosity)
              }
    
    # Sample from the posterior distribution
    os.makedirs("ABCSubSim_output")
    os.chdir("ABCSubSim_output")
    
    if enableInteriorViscosity:
        if not checkpointed:
            ABCSubSim_sampler = ABCSubSim.ABCSubSim(problem,logpath="%s/ABCSubSim_Hemocell_visc_%i_%i_tmax_%i_log.pkl" % (outputpath,imin,imax,tmax), 
                                                    logstep=1000, tol=0.02, invPa=10, max_stages=5, nprocs=nprocs)
        else:
            ABCSubSim_sampler = ABCSubSim.load_state("%s/ABCSubSim_Hemocell_visc_%i_%i_tmax_%i_log.pkl" % (outputpath,imin,imax,tmax))
    else:
        if not checkpointed:
            ABCSubSim_sampler = ABCSubSim.ABCSubSim(problem,logpath="%s/ABCSubSim_Hemocell_normal_%i_%i_tmax_%i_log.pkl" % (outputpath,imin,imax,tmax), 
                                                    logstep=1000, tol=0.02, invPa=10, max_stages=5, nprocs=nprocs)
        else:
            ABCSubSim_sampler = ABCSubSim.load_state("%s/ABCSubSim_Hemocell_normal_%i_%i_tmax_%i_log.pkl" % (outputpath,imin,imax,tmax))

    df,qoi = ABCSubSim_sampler.sample(n_samples,checkpoint=checkpointed)
    
    os.chdir("..")

    # Remove garbage
    #shutil.rmtree("./ABCSubSim_output")
    
    if enableInteriorViscosity:
        df.to_csv("ABCSubSim_hemocell_samples_visc_%i_%i_tmax_%i.csv" % (imin,imax,tmax),sep=";",index=False)
        np.save("ABCSubSim_hemocell_qoi_visc_%i_%i_tmax_%i.npy" % (imin,imax,tmax),qoi)
    else:
        df.to_csv("ABCSubSim_hemocell_samples_normal_%i_%i_tmax_%i.csv" % (imin,imax,tmax),sep=";",index=False)
        np.save("ABCSubSim_hemocell_qoi_normal_%i_%i_tmax_%i.npy" % (imin,imax,tmax),qoi)
