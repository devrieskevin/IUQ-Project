import sys
import os
import shutil

import argparse

import numpy as np
from scipy.stats import norm,uniform,multivariate_normal

import pandas as pd
import dill

import hemocell.model as hemocell

from lxml import etree

from pyDOE import lhs

#from local_config import *
#from lisa_config import *
from cartesius_config import *

from run_tools import run_external

np.random.seed(676767)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--nsamples",dest="nsamples",type=int,default=1000)
    parser.add_argument("--imin",dest="imin",type=int,default=3)
    parser.add_argument("--imax",dest="imax",type=int,default=12)
    parser.add_argument("--enableInteriorViscosity",dest="enableInteriorViscosity",type=int,default=0)
    parser.add_argument("--nprocs",dest="nprocs",type=int,default=16)
    parser.add_argument("--model_type",dest="model_type",type=str,default="external")

    args = parser.parse_args()

    # Set design variable argument values
    nsamples = args.nsamples
    imin = args.imin
    imax = args.imax
    enableInteriorViscosity = args.enableInteriorViscosity
    nprocs = args.nprocs
    model_type = args.model_type

    if enableInteriorViscosity:
        model_params = ["kLink","kBend","viscosityRatio","shearrate","tmax","tstart","tmeas"]
    else:
        model_params = ["kLink","kBend","shearrate","tmax","tstart","tmeas"]

    design_vars = ["enableInteriorViscosity"]

    # Extract data from dataset
    data = pd.read_csv("%s/Ekcta_full.csv" % (datapath),sep=";")
    data = data.loc[data["Treatment"] == 0]
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

    design_vals = np.array([enableInteriorViscosity])

    # Construct problem dict
    problem = {"model_type":model_type,
               "setup":(lambda params: hemocell.setup(modelpath,params)),
               "measure":hemocell.measureEI_convergence,
               "params":model_params,
               "design_vars":design_vars,
               "input_data":design_vals
              }

    # Set the bounds on the parameters
    bounds = [[10.0,300.0],
              [50.0,400.0]]

    if enableInteriorViscosity:
        bounds.append([1.0,15.0])

    samples = np.zeros((stress.size*nsamples,len(model_params)))
    for n,gamma in enumerate(shearrate):
        hsamples = lhs(len(model_params)-4,samples=nsamples,criterion="maximin")

        for m,(bmin,bmax) in enumerate(bounds):
            hsamples[:,m] = bmin + (bmax-bmin)*hsamples[:,m]

        samples[n*nsamples:(n+1)*nsamples,:len(model_params)-4] = hsamples
        samples[n*nsamples:(n+1)*nsamples,len(model_params)-4] = gamma
        samples[n*nsamples:(n+1)*nsamples,len(model_params)-3] = tmax[n]
        samples[n*nsamples:(n+1)*nsamples,len(model_params)-2] = tstart[n]
        samples[n*nsamples:(n+1)*nsamples,len(model_params)-1] = tmeas[n]

    qoi, c_err = run_external(problem,samples,nprocs=nprocs,path="train_output",keep_output=False)

    # Write output to files
    if enableInteriorViscosity:
        np.save("%s/train_hemocell_samples_visc_%i_%i_nsamples_%i.npy" % (outputpath,imin,imax,nsamples),samples)
        np.save("%s/train_hemocell_qoi_visc_%i_%i_nsamples_%i.npy" % (outputpath,imin,imax,nsamples),qoi)
        np.save("%s/train_hemocell_c_err_visc_%i_%i_nsamples_%i.npy" % (outputpath,imin,imax,nsamples),c_err)
    else:
        np.save("%s/train_hemocell_samples_normal_%i_%i_nsamples_%i.npy" % (outputpath,imin,imax,nsamples),samples)
        np.save("%s/train_hemocell_qoi_normal_%i_%i_nsamples_%i.npy" % (outputpath,imin,imax,nsamples),qoi)
        np.save("%s/train_hemocell_c_err_normal_%i_%i_nsamples_%i.npy" % (outputpath,imin,imax,nsamples),c_err)
