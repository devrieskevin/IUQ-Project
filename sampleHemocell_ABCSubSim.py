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
#from cartesius_config import *

# Set seed for reproducibility
np.random.seed(77777)

def distance(a,b):
    return np.sqrt(np.mean((a-b)**2))

def model_priors(enableInteriorViscosity):
    kLink_prior = lambda sample: uniform.pdf(sample,10.0,290.0)
    kBend_prior = lambda sample: uniform.pdf(sample,50.0,350.0)
    viscosityRatio_prior = lambda sample: uniform.pdf(sample,1.0,14.0)
    
    if enableInteriorViscosity:
        return [kLink_prior,kBend_prior,viscosityRatio_prior]
    else:
        return [kLink_prior,kBend_prior]

def model_samplers(enableInteriorViscosity):
    kLink_sampler = lambda n_samples: np.random.uniform(10.0,300.0,n_samples)
    kBend_sampler = lambda n_samples: np.random.uniform(50.0,400.0,n_samples)
    viscosityRatio_sampler = lambda n_samples: np.random.uniform(1.0,15.0,n_samples)
    
    if enableInteriorViscosity:
        return [kLink_sampler,kBend_sampler,viscosityRatio_sampler]
    else:
        return [kLink_sampler,kBend_sampler]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--n_samples",dest="n_samples",type=int,default=100)
    parser.add_argument("--enableInteriorViscosity",type=int,default=0)
    parser.add_argument("--checkpointed",dest="checkpointed",action="store_true",default=False)
    parser.add_argument("--imin",dest="imin",type=int,default=3)
    parser.add_argument("--imax",dest="imax",type=int,default=12)
    parser.add_argument("--nprocs",dest="nprocs",type=int,default=16)
    parser.add_argument("--model_type",dest="model_type",type=str,default="external")

    args = parser.parse_args()

    # Set design variable argument values
    n_samples = args.n_samples
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

    design_vars = ["shearrate","tmax","tmeas","tgamma","enableInteriorViscosity"]
    
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

    # Determine the number of timesteps and measurements
    tgamma = 7.5
    tconv = np.array([int(tgamma / (0.5e-7 * gamma)) for gamma in shearrate])
    tmax = np.ceil(tconv * 1.25)
    tmeas = 2000

    design_vals = np.column_stack(np.broadcast_arrays(shearrate,tmax,tmeas,tgamma,enableInteriorViscosity))

    # Construct problem dict
    problem = {"model_type":model_type,
               "setup":(lambda params: hemocell.setup(modelpath,params)),
               "measure":hemocell.measureEI_convergence,
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
            ABCSubSim_sampler = ABCSubSim.ABCSubSim(problem,logpath="%s/ABCSubSim_Hemocell_visc_%i_%i_log.pkl" % (outputpath,imin,imax), 
                                                    logstep=2000, tol=0.02, invPa=10, max_stages=5, nprocs=nprocs)
        else:
            ABCSubSim_sampler = ABCSubSim.load_state("%s/ABCSubSim_Hemocell_visc_%i_%i_log.pkl" % (outputpath,imin,imax))
    else:
        if not checkpointed:
            ABCSubSim_sampler = ABCSubSim.ABCSubSim(problem,logpath="%s/ABCSubSim_Hemocell_normal_%i_%i_log.pkl" % (outputpath,imin,imax), 
                                                    logstep=2000, tol=0.02, invPa=10, max_stages=5, nprocs=nprocs)
        else:
            ABCSubSim_sampler = ABCSubSim.load_state("%s/ABCSubSim_Hemocell_normal_%i_%i_log.pkl" % (outputpath,imin,imax))

    df,qoi,c_err = ABCSubSim_sampler.sample(n_samples,checkpoint=checkpointed)
    
    os.chdir("..")

    # Remove garbage
    #shutil.rmtree("./ABCSubSim_output")
    
    if enableInteriorViscosity:
        df.to_csv("ABCSubSim_hemocell_samples_visc_%i_%i.csv" % (imin,imax),sep=";",index=False)
        np.save("ABCSubSim_hemocell_qoi_visc_%i_%i.npy" % (imin,imax),qoi)
        np.save("ABCSubSim_hemocell_c_err_visc_%i_%i.npy" % (imin,imax),c_err)
    else:
        df.to_csv("ABCSubSim_hemocell_samples_normal_%i_%i.csv" % (imin,imax),sep=";",index=False)
        np.save("ABCSubSim_hemocell_qoi_normal_%i_%i.npy" % (imin,imax),qoi)
        np.save("ABCSubSim_hemocell_c_err_normal_%i_%i.npy" % (imin,imax),c_err)
