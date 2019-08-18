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
    parser.add_argument("--index",dest="index",type=int)
    args = parser.parse_args()

    # Set design variable argument values
    index = args.index

    model_params = ["kLink","kBend","viscosityRatio"]
    design_vars = ["shearrate","tmax","tstart","tmeas"]

    # Extract data from dataset
    data = pd.read_csv("%s/Ekcta_100.csv" % (datapath),sep=";")
    data = data.loc[data["Treatment"] == 0.5]
    stress,el,el_err = data.values[index,[1,3,4]].T

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
    tstart = tgamma // (0.5e-7 * shearrate)
    tmax = np.ceil(tstart * 1.25)
    tmeas = (tmax-tstart) // 10

    design_vals = np.column_stack(np.broadcast_arrays(shearrate,tmax,tstart,tmeas))

    # Construct problem dict
    problem = {"model_type":"external",
               "setup":(lambda params: hemocell.setup(modelpath,params)),
               "measure":hemocell.measureEI_convergence,
               "params":model_params,
               "design_vars":design_vars,
               "input_data":design_vals
              }
    
    samples = np.array([120,275,5.2])[None,:]
    qoi, c_err = run_external(problem,samples,nprocs=1,path="sample_output_%i" % (index))

    print("Shear stress:",stress)
    print("Shear rate:",shearrate)
