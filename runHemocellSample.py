import sys
import os
import shutil

import argparse

import numpy as np
import pandas as pd

import hemocell.model as hemocell

from lxml import etree

#from local_config import *
#from lisa_config import *
from cartesius_config import *

from run_tools import run_external

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--imin",dest="imin",type=int,default=3)
    parser.add_argument("--imax",dest="imax",type=int,default=12)
    parser.add_argument("--enableInteriorViscosity",dest="enableInteriorViscosity",type=int,default=0)
    parser.add_argument("--model_type",dest="model_type",type=str,default="external")
    parser.add_argument("--model",dest="model",type=str,default="hemocell")
    parser.add_argument("--errType",dest="errType",type=str,default="EL_error")
    parser.add_argument("--method",dest="method",type=str,default="TMCMC")
    parser.add_argument("--cellHealth",dest="cellHealth",type=str,default="healthy")
    parser.add_argument("--lmax",dest="lmax",type=int,default=1)
    parser.add_argument("--nburn",dest="nburn",type=int,default=0)
    parser.add_argument("--nsamples",dest="nsamples",type=int,default=10000)

    args = parser.parse_args()

    # Set design variable argument values
    imin = args.imin
    imax = args.imax
    enableInteriorViscosity = args.enableInteriorViscosity
    model_type = args.model_type
    model = args.model
    errType = args.errType
    method = args.method
    cellHealth = args.cellHealth
    lmax = args.lmax
    nburn = args.nburn
    nsamples = args.nsamples

    if enableInteriorViscosity:
        model_params = ["kLink","kBend","viscosityRatio"]
    else:
        model_params = ["kLink","kBend"]

    design_vars = ["enableInteriorViscosity","shearrate","tmax","tstart","tmeas"]

    # Extract data from dataset
    data = pd.read_csv("%s/Ekcta_full.csv" % (datapath),sep=";")
    data = data.loc[data["Treatment"] == 0.5]
    stress,el,el_err = data.values[imin:imax,[1,2,3]].T

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

    design_vals = np.column_stack(np.broadcast_arrays(enableInteriorViscosity,shearrate,tmax,tstart,tmeas))

    # Construct problem dict
    problem = {"model_type":model_type,
               "setup":(lambda params: hemocell.setup(modelpath,params)),
               "measure":hemocell.measureEI_convergence,
               "params":model_params,
               "design_vars":design_vars,
               "input_data":design_vals
              }

    if enableInteriorViscosity:
        mode = "visc"
    else:
        mode = "normal"
            
    if method == "TMCMC":
        filename = "%s/%s_%s_%s_%s_samples_%s_%i_%i_lmax_%s_nburn_%i_nsamples_%i.csv" % (outputpath,method,model,cellHealth,
                                                                                         errType,mode,imin,imax,lmax,nburn,nsamples)
        print("File name sample file:",filename)
        sample_df = pd.read_csv(filename,sep=";")
    
    mpe = np.argmax(sample_df["likelihood"].values * sample_df["prior"].values)
    samples = sample_df.loc[mpe][["kLink","kBend","viscosityRatio"]].values[None,:]
    
    qoi, c_err = run_external(problem,samples,nprocs=24,path="sample_output")

    if model != "hemocell":
        np.save("%s/%s_%s_%s_%s_qoi_%s_%i_%i_lmax_%s_nburn_%i_nsamples_%i_mpe_sample.npy" % 
                (outputpath,method,model,cellHealth,errType,mode,imin,imax,lmax,nburn,nsamples),qoi)
    
        np.save("%s/%s_%s_%s_%s_c_err_%s_%i_%i_lmax_%s_nburn_%i_nsamples_%i_mpe_sample.npy" % 
                (outputpath,method,model,cellHealth,errType,mode,imin,imax,lmax,nburn,nsamples),c_err)
