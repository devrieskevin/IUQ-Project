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
from lisa_config import *
#from cartesius_config import *

from run_tools import run_external

np.random.seed(676767)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--n_samples",dest="n_samples",type=int,default=1000)
    parser.add_argument("--imin",dest="imin",type=int,default=3)
    parser.add_argument("--imax",dest="imax",type=int,default=12)
    parser.add_argument("--enableInteriorViscosity",dest="enableInteriorViscosity",type=int,default=0)
    parser.add_argument("--nprocs",dest="nprocs",type=int,default=16)
    parser.add_argument("--model_type",dest="model_type",type=str,default="external")

    args = parser.parse_args()

    # Set design variable argument values
    n_samples = args.n_samples
    imin = args.imin
    imax = args.imax
    enableInteriorViscosity = args.enableInteriorViscosity
    nprocs = args.nprocs
    model_type = args.model_type

    if enableInteriorViscosity:
        model_params = ["kLink","kBend","viscosityRatio","shearrate","tmax"]
    else:
        model_params = ["kLink","kBend","shearrate","tmax"]

    design_vars = ["tmeas","tgamma","enableInteriorViscosity"]

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

    design_vals = np.column_stack(np.broadcast_arrays(tmeas,tgamma,enableInteriorViscosity))

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

    samples = np.zeros((stress.size*n_samples,len(model_params)))
    for n,gamma in enumerate(shearrate):
        hsamples = lhs(len(model_params)-2,samples=n_samples,criterion="maximin")

        for m,(bmin,bmax) in enumerate(bounds):
            hsamples[:,m] = bmin + (bmax-bmin)*hsamples[:,m]

        samples[n*n_samples:(n+1)*n_samples,:len(model_params)-2] = hsamples
        samples[n*n_samples:(n+1)*n_samples,len(model_params)-2] = gamma
        samples[n*n_samples:(n+1)*n_samples,len(model_params)-1] = tmax[n]

    os.makedirs("train_output")
    os.chdir("train_output")

    qoi, c_err = run_external(problem,samples,nprocs=nprocs)

    os.chdir("..")

    # Write output to files
    if enableInteriorViscosity:
        np.save("train_hemocell_samples_visc_%i_%i.npy" % (imin,imax),samples)
        np.save("train_hemocell_qoi_visc_%i_%i.npy" % (imin,imax),qoi)
        np.save("train_hemocell_c_err_visc_%i_%i.npy" % (imin,imax),c_err)
    else:
        np.save("train_hemocell_samples_normal_%i_%i.npy" % (imin,imax),samples)
        np.save("train_hemocell_qoi_normal_%i_%i.npy" % (imin,imax),qoi)
        np.save("train_hemocell_c_err_normal_%i_%i.npy" % (imin,imax),c_err)
