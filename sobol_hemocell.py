import os
import shutil

import argparse

import numpy as np
from scipy.stats import norm,uniform,multivariate_normal

import pandas as pd
import dill

import hemocell.model as hemocell

from lxml import etree

from SALib.sample import saltelli

#from local_config import *
#from lisa_config import *
from cartesius_config import *

from run_tools import run_external


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--n_samples",dest="n_samples",type=int,default=1000)
    parser.add_argument("--imin",dest="imin",type=int,default=3)
    parser.add_argument("--imax",dest="imax",type=int,default=12)
    parser.add_argument("--tmax",dest="tmax",type=int,required=True)
    parser.add_argument("--enableInteriorViscosity",dest="enableInteriorViscosity",type=int,default=0)
    parser.add_argument("--nprocs",dest="nprocs",type=int,default=16)
    parser.add_argument("--model_type",dest="model_type",type=str,default="external")

    args = parser.parse_args()

    # Set design variable argument values
    n_samples = args.n_samples
    imin = args.imin
    imax = args.imax
    tmax = args.tmax
    enableInteriorViscosity = args.enableInteriorViscosity
    nprocs = args.nprocs
    model_type = args.model_type

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

    design_vals = np.column_stack(np.broadcast_arrays(shearrate,tmax,tmax,enableInteriorViscosity))

    # Construct problem dict
    problem = {"model_type":model_type,
               "setup":(lambda params: hemocell.setup(modelpath,params)),
               "measure":hemocell.measureEI,
               "params":model_params,
               "design_vars":design_vars,
               "input_data":design_vals
              }

    # Set the bounds on the parameters
    bounds = [[15.0,300.0],
              [80.0,400.0]]

    if enableInteriorViscosity:
        bounds.append([1.0,50.0])

    # Problem for Sobol analysis
    sobol_problem = {"num_vars":3,
                     "names":model_params,
                     "bounds":bounds
                    }

    param_values = saltelli.sample(sobol_problem,n_samples,calc_second_order=False)

    # Sample from the posterior distribution
    os.makedirs("sobol_output")
    os.chdir("sobol_output")

    qoi, c_err = run_external(problem,param_values,nprocs=nprocs)

    os.chdir("..")

    # Write output to files
    if enableInteriorViscosity:
        np.save("sobol_hemocell_qoi_visc_%i_%i_tmax_%i.npy" % (imin,imax,tmax),qoi)
    else:
        np.save("sobol_hemocell_qoi_normal_%i_%i_tmax_%i.npy" % (imin,imax,tmax),qoi)
