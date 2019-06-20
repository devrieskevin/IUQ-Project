import os
import shutil

import argparse

import numpy as np
import pandas as pd

import hemocell.model as hemocell
import run_tools

from lxml import etree

#from local_config import *
#from lisa_config import *
from cartesius_config import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--tmax",dest="tmax",type=int,required=True)
    parser.add_argument("--tmeas",dest="tmeas",type=int,required=True)
    parser.add_argument("--enableInteriorViscosity",dest="enableInteriorViscosity",type=int,default=0)
    parser.add_argument("--imin",dest="imin",type=int,default=0)
    parser.add_argument("--imax",dest="imax",type=int,default=10)
    parser.add_argument("--kLink",dest="kLink",type=float,default=15)
    parser.add_argument("--kBend",dest="kBend",type=float,default=80)
    parser.add_argument("--viscosityRatio",dest="viscosityRatio",type=float,default=5)

    args = parser.parse_args()

    # Set design variable argument values
    tmax = args.tmax
    tmeas = args.tmeas
    enableInteriorViscosity = args.enableInteriorViscosity
    imin = args.imin
    imax = args.imax
    kLink = args.kLink
    kBend = args.kBend
    viscosityRatio = args.viscosityRatio

    params = ["kLink","kBend","viscosityRatio"]
    design_vars = ["shearrate","enableInteriorViscosity","tmax","tmeas"]
    
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
    design_vals = np.column_stack(np.broadcast_arrays(shearrate,enableInteriorViscosity,tmax,tmeas))

    # Single
    sample = [kLink,kBend,viscosityRatio]

    # Construct problem dict
    problem = {"setup":(lambda params: hemocell.setup(modelpath,params)),
               "measure":hemocell.measureEI_full,
               "params":params,
               "design_vars":design_vars,
               "input_data":design_vals
              }
    
    qoi, c_err = run_tools.run_batch(problem,sample,nprocs=24)

    # Write output to files
    if enableInteriorViscosity:
        np.save("convergence_qoi_visc_%i_%i_%i_%i_%i.npy" % (imin,imax,kLink,kBend,viscosityRatio),qoi)
    else:
        np.save("convergence_qoi_normal_%i_%i_%i_%i_%i.npy" % (imin,imax,kLink,kBend,viscosityRatio),qoi)

