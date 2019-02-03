import numpy as np
import sys

sys.path.append("/home/kevin/master_project/IUQ-Project/measure")
import HCELL_readhdf5
import EL

from lxml import etree

def buildMaterialXml(filename,params,name="RBC",comment="Parameters for the HO RBC constitutive model."):
    root = etree.Element("hemocell")
    material = etree.SubElement(root,"MaterialModel")

    etree.SubElement(material,"comment").text = "Parameters for the HO RBC constitutive model."
    etree.SubElement(material,"name").text = "RBC"

    # Add material parameters
    for key in params:
        etree.SubElement(material,key).text = str(params[key])

    # Build xml tree
    tree = etree.ElementTree(root)
    tree.write(filename,xml_declaration=True,pretty_print=True)
    
    return


TIME_BEGIN = 98000
TIME_END = 100000
timestep = 2000
nprocs = 1
read_procs = 1


datapath = "/home/kevin/master_project/HemoCell/examples/oneCellShear/tmp/hdf5/"

for t in range(TIME_BEGIN,TIME_END,timestep):
    fluid,rbc,platelet,ct3 = HCELL_readhdf5.open_hdf5_files(p=False,f=False,ct3=False,half=True,read_procs=read_procs,nprocs=nprocs,begin=t,end=t+timestep,timestep=timestep,datapath=datapath)

    #Measure elongation index
    pos = np.array(rbc[0].position)
    X = pos[:,0]
    Y = pos[:,1]
    Z = pos[:,2]

    A,B,elongation_index = EL.elongation_index(X,Y)

    experimental_curves = EL.get_mario_curves(datapath = "/home/kevin/master_project/data/")

    print(elongation_index)

params = {}

params["eta_m"] = 0.0
params["kBend"] = 80.0
params["kVolume"] = 20.0
params["kArea"] = 5.0
params["kLink"] = 15.0
params["minNumTriangles"] = 600
params["radius"] = 3.91e-6
params["Volume"] = 90

buildMaterialXml("testfile.xml",params)
