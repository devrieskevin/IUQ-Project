import numpy as np
import sys

sys.path.append("/home/kevin/master_project/IUQ-Project/measure")
import HCELL_readhdf5
import EL

from lxml import etree

def buildMaterialXml(source,kLink,kBend,viscosityRatio,dest=None):
    tree = etree.parse(source,parser=etree.XMLParser(remove_blank_text=True,remove_comments=True))
    root = tree.getroot()

    # Add material parameters
    elements = root.findall('MaterialModel/*')

    for elem in elements:
        if elem.tag == 'kLink':
            elem.text = " " + str(kLink) + " "

        if elem.tag == 'kBend':
            elem.text = " " + str(kBend) + " "

        if elem.tag == 'viscosityRatio':
            elem.text = " " + str(viscosityRatio) + " "

    if dest:
        tree.write(dest,xml_declaration=True,pretty_print=True)
    else:
        tree.write(source,xml_declaration=True,pretty_print=True)

    return


def buildConfig(source,rateval,outDir=None,dest=None):
    tree = etree.parse(source,parser=etree.XMLParser(remove_blank_text=True,remove_comments=True))
    root = tree.getroot()

    # Find relevant elements
    shearrate = root.findall('domain/shearrate')[0]
    parameters = root.findall('parameters')[0]

    # Change shear rate
    shearrate.text = " " + str(rateval) + " "
    
    # Specify output directory
    if outDir:
        output = parameters.findall('outputDirectory')
        if not output:
            output = etree.SubElement(parameters,'outputDirectory')
        else:
            output = output[0]

        output.text = outDir

    if dest:
        tree.write(dest,xml_declaration=True,pretty_print=True)
    else:
        tree.write(source,xml_declaration=True,pretty_print=True)

    return

buildMaterialXml("/home/kevin/master_project/IUQ-Project/measure/xml_templates/RBC_HO_template.xml",1,1,1,dest="/home/kevin/master_project/HemoCell/examples/oneCellShear/RBC_HO.xml")
buildConfig("/home/kevin/master_project/IUQ-Project/measure/xml_templates/config_template.xml",50.0,dest="/home/kevin/master_project/HemoCell/examples/oneCellShear/config.xml")

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

