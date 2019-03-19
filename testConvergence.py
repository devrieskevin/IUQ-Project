import os
import shutil

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
    # Define problem parameters
    params = ["kLink","kBend"]
    design_vars = ["shearrate"]
    
    # Extract data from dataset
    data = pd.read_csv("%s/Ekcta_100.csv" % (datapath),sep=";")
    data = data.loc[data["Treatment"] == 0.5]
    stress,el,el_err = data.values[3:12,[1,3,4]].T

    # Get data from config files
    configpath = "%s/hemocell/templates/config_template.xml" % (libpath)
    tree = etree.parse(configpath, parser=etree.XMLParser(remove_blank_text=True, remove_comments=True))
    root = tree.getroot()
    nuP = float(root.find("domain/nuP").text)
    rhoP = float(root.find("domain/rhoP").text)

    # Compute the shear rates
    shearrate = stress / (nuP * rhoP)
    design_vals = shearrate

    # Construct sample array
    df = pd.read_csv("TMCMC_hemocell_samples_normal_3_12.csv",sep=";")
    df = df.loc[np.argmax(df["likelihood"].values)]
    model_params = df.loc[params].values

    tmax = 100000
    tmeas = 2000

    t_vals = np.arange(tmeas,tmax+1,tmeas)

    tiled = np.tile(model_params,(t_vals.shape[0],1))
    samples = np.column_stack([tiled,t_vals,t_vals])
    #sample = model_params

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
    np.save("convergence_qoi.npy",qoi)
