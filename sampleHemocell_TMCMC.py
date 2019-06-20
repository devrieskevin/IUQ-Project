import os
import shutil

import argparse

import numpy as np
from scipy.stats import norm,uniform,multivariate_normal

import pandas as pd
import dill

import UQLib.calibration.TMCMC as TMCMC
import hemocell.model as hemocell

from lxml import etree

#from local_config import *
#from lisa_config import *
from cartesius_config import *

# Set seed for reproducibility
np.random.seed(7773777)

def model_prior(sample,enableInteriorViscosity):
    kLink_prior = uniform.pdf(sample[0],10.0,290.0)
    kBend_prior = uniform.pdf(sample[1],50.0,350.0)
    
    if enableInteriorViscosity:
        viscosityRatio_prior = uniform.pdf(sample[2],1.0,14.0)
        return np.prod([kLink_prior,kBend_prior,viscosityRatio_prior])
    else:
        return np.prod([kLink_prior,kBend_prior])

def model_sampler(n_samples,enableInteriorViscosity):
    kLink_samples = np.random.uniform(10.0,300.0,n_samples)
    kBend_samples = np.random.uniform(50.0,400.0,n_samples)
    
    if enableInteriorViscosity:
        viscosityRatio_samples = np.random.uniform(1.0,15.0,n_samples)
        return np.column_stack([kLink_samples,kBend_samples,viscosityRatio_samples])
    else:
        return np.column_stack([kLink_samples,kBend_samples])

def error_prior(sample):
    return np.prod(uniform.pdf(sample,0.001,0.099))

def error_sampler(n_samples):
    return np.random.uniform(0.001,0.1,(n_samples,1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--n_samples",dest="n_samples",type=int,default=100)
    parser.add_argument("--lmax",dest="lmax",type=str,default="inf")
    parser.add_argument("--treatment",dest="treatment",type=float,default=0)
    parser.add_argument("--enableInteriorViscosity",dest="enableInteriorViscosity",type=int,default=0)
    parser.add_argument("--checkpointed",dest="checkpointed",action="store_true",default=False)
    parser.add_argument("--imin",dest="imin",type=int,default=3)
    parser.add_argument("--imax",dest="imax",type=int,default=12)
    parser.add_argument("--nprocs",dest="nprocs",type=int,default=16)
    parser.add_argument("--model_type",dest="model_type",type=str,default="external")

    args = parser.parse_args()

    # Set design variable argument values
    n_samples = args.n_samples

    lmax = args.lmax
    if lmax == "inf":
        lmax = np.inf
    elif lmax.isdigit():
        lmax = int(lmax)

    treatment = args.treatment
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
        
    error_params = ["err"]
    design_vars = ["shearrate","tmax","tmeas","tstart","enableInteriorViscosity"]

    # Extract data from dataset
    data = pd.read_csv("%s/Ekcta_100.csv" % (datapath),sep=";")

    if treatment == 0:
        data = data.loc[data["Treatment"] == 0.5]
        stress,el = data.values[imin:imax,[1,2]].T
        el_err = np.zeros(el.shape)
        cellHealth = "healthy"
    elif treatment > 0:
        data = data.loc[data["Treatment"] == treatment]
        stress,el,el_err = data.values[imin:imax,[1,3,4]].T
        cellHealth = "treated"

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
    tstart = (tgamma / (0.5e-7 * shearrate)).astype(int)
    tmax = np.ceil(tstart * 1.25)
    tmeas = ((tmax-tstart) / 10).astype(int)

    design_vals = np.column_stack(np.broadcast_arrays(shearrate,tmax,tmeas,tstart,enableInteriorViscosity))

    # Map model errors to data samples
    error_mapping = ["err" for n in range(shearrate.shape[0])]

    # Construct problem dict
    problem = {"model_type":model_type,
               "setup":(lambda params: hemocell.setup(modelpath,params)),
               "measure":hemocell.measureEI_convergence,
               "model_params":model_params,
               "error_params":error_params,
               "design_vars":design_vars,
               "input_data":design_vals,
               "output_data":el,
               "data_errors":el_err,
               "error_mapping":error_mapping,
               "model_prior":(lambda sample: model_prior(sample,enableInteriorViscosity)),
               "model_sampler":(lambda n_samples: model_sampler(n_samples,enableInteriorViscosity)),
               "error_prior":error_prior,
               "error_sampler":error_sampler
              }

    if enableInteriorViscosity:
        if not checkpointed:
            TMCMC_sampler = TMCMC.TMCMC(problem,lmax=lmax,logpath="%s/TMCMC_hemocell_%s_visc_%i_%i_lmax_%s_log.pkl" % (outputpath,cellHealth,imin,imax,lmax),logstep=2000,nprocs=nprocs)
        else:
            TMCMC_sampler = TMCMC.load_state("%s/TMCMC_hemocell_%s_visc_%i_%i_lmax_%s_log.pkl" % (outputpath,cellHealth,imin,imax,lmax))
    else:
        if not checkpointed:
            TMCMC_sampler = TMCMC.TMCMC(problem,lmax=lmax,logpath="%s/TMCMC_hemocell_%s_normal_%i_%i_lmax_%s_log.pkl" % (outputpath,cellHealth,imin,imax,lmax),logstep=2000,nprocs=nprocs)
        else:
            TMCMC_sampler = TMCMC.load_state("%s/TMCMC_hemocell_%s_normal_%i_%i_lmax_%s_log.pkl" % (outputpath,cellHealth,imin,imax,lmax))

    df,qoi,c_err = TMCMC_sampler.sample(n_samples,checkpoint=checkpointed)

    # Write output to files
    if enableInteriorViscosity:
        df.to_csv("TMCMC_hemocell_%s_samples_visc_%i_%i_lmax_%s.csv" % (cellHealth,imin,imax,lmax),sep=";",index=False)
        np.save("TMCMC_hemocell_%s_qoi_visc_%i_%i_lmax_%s.npy" % (cellHealth,imin,imax,lmax),qoi)
        np.save("TMCMC_hemocell_%s_c_err_visc_%i_%i_lmax_%s.npy" % (cellHealth,imin,imax,lmax),c_err)
    else:
        df.to_csv("TMCMC_hemocell_%s_samples_normal_%i_%i_lmax_%s.csv" % (cellHealth,imin,imax,lmax),sep=";",index=False)
        np.save("TMCMC_hemocell_%s_qoi_normal_%i_%i_lmax_%s.npy" % (cellHealth,imin,imax,lmax),qoi)
        np.save("TMCMC_hemocell_%s_c_err_normal_%i_%i_lmax_%s.npy" % (cellHealth,imin,imax,lmax),c_err)
