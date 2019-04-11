import os
import shutil

import argparse

import numpy as np
from scipy.stats import norm,uniform,multivariate_normal

import pandas as pd

import UQLib.calibration.TMCMC as TMCMC
import hemocell.model as hemocell
import run_tools

from lxml import etree

#from local_config import *
from lisa_config import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--method",dest="method",type=str,default="TMCMC")
    parser.add_argument("--tmax",dest="tmax",type=int,required=True)
    parser.add_argument("--tmeas",dest="tmeas",type=int,required=True)
    parser.add_argument("--tsource",dest="tsource",type=int,required=True)
    parser.add_argument("--enableInteriorViscosity",dest="enableInteriorViscosity",type=int,default=0)
    parser.add_argument("--imin",dest="imin",type=int,default=3)
    parser.add_argument("--imax",dest="imax",type=int,default=12)

    args = parser.parse_args()

    # Set design variable argument values
    method = args.method
    tsource = args.tsource
    tmax = args.tmax
    tmeas = args.tmeas
    enableInteriorViscosity = args.enableInteriorViscosity
    imin = args.imin
    imax = args.imax

    # Define problem parameters
    if enableInteriorViscosity:
        params = ["kLink","kBend","viscosityRatio"]
    else:
        params = ["kLink","kBend"]
    
    design_vars = ["shearrate","enableInteriorViscosity"]
    
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
    design_vals = np.row_stack(np.broadcast(shearrate,enableInteriorViscosity))

    # Construct sample array
    if enableInteriorViscosity:
        df = pd.read_csv("%s/%s_hemocell_samples_visc_%i_%i_tmax_%i.csv" % (outputpath,method,imin,imax,tsource),sep=";")
    else:
        df = pd.read_csv("%s/%s_hemocell_samples_normal_%i_%i_tmax_%i.csv" % (outputpath,method,imin,imax,tsource),sep=";")
    
    if method == "TMCMC":
        df = df.loc[np.argmax(df["likelihood"].values)]
    elif method == "ABCSubSim":
        df = df.loc[np.argmin(df["distance"].values)]

    model_params = df.loc[params].values

    t_vals = np.arange(tmeas,tmax+1,tmeas)

    tiled = np.tile(model_params,(t_vals.shape[0],1))
    samples = np.column_stack([tiled,t_vals,t_vals])

    # Construct problem dict
    problem = {"setup":(lambda params: hemocell.setup(modelpath,params)),
               "measure":hemocell.measureEI,
               "params":params + ["tmax","tmeas"],
               "design_vars":design_vars,
               "input_data":design_vals
              }
    
    # Sample from the posterior distribution
    os.makedirs("Convergence_output")
    os.chdir("Convergence_output")
    
    #qoi, c_err = run_tools.run_batch(problem,sample,tmax,tmeas,nprocs=16)
    qoi, c_err = run_tools.run_external(problem,samples,nprocs=16)
    
    os.chdir("..")

    # Write output to files
    if enableInteriorViscosity:
        np.save("convergence_qoi_visc_%i_%i.npy" % (imin,imax),qoi)
    else:
        np.save("convergence_qoi_normal_%i_%i.npy" % (imin,imax),qoi)
