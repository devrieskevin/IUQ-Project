import os
import shutil

import numpy as np

# Local modules for data reading and measurement
from . import HCELL_readhdf5
from . import EL

# Import XML parser
from lxml import etree

def setup(modelpath, params):
    """
    Implements the setup for a run of the Hemocell model
    Returns the list of command line arguments
    """

    templatepath = "%s/templates" % (os.path.dirname(__file__))

    # Build and copy the necessary files
    buildMaterialXml("%s/RBC_HO_template.xml" % (templatepath), params, dest="RBC_HO.xml")
    buildConfigXml("%s/config_template.xml" % (templatepath), params, dest="config.xml")
    shutil.copyfile("%s/RBC_HO.pos" % (templatepath), "./RBC_HO.pos")

    return [modelpath, "config.xml"]


def buildMaterialXml(source, params, dest=None):
    """
    Builds an XML file for the RBC material model by reading in a RBC XML template and setting
    the relevant parameters: kLink, kBend and the viscosity ratio
    """

    tree = etree.parse(source, parser=etree.XMLParser(remove_blank_text=True, remove_comments=True))
    root = tree.getroot()

    for name in params.keys():
        elem = root.find('MaterialModel/%s' % name)
        if elem is not None:
            if name == "enableInteriorViscosity":
                elem.text = " " + str(int(params[name])) + " "
            else:
                elem.text = " " + str(params[name]) + " "

    if dest:
        tree.write(dest, xml_declaration=True, pretty_print=True)
    else:
        tree.write(source, xml_declaration=True, pretty_print=True)

    return


def buildConfigXml(source, params, outDir=None, dest=None):
    """
    Builds a config XML file by reading in a config template and setting the shear rate 
    """

    tree = etree.parse(source, parser=etree.XMLParser(remove_blank_text=True, remove_comments=True))
    root = tree.getroot()

    # Find relevant elements
    shearrate = root.find('domain/shearrate')
    parameters = root.find('parameters')

    # Change shear rate
    shearrate.text = " " + str(params["shearrate"]) + " "
    
    # Specify output directory
    if outDir:
        output = parameters.find('outputDirectory')
        if not output:
            output = etree.SubElement(parameters, 'outputDirectory')
        else:
            output = output

        output.text = outDir

    for name in params.keys():
        elem = root.find('sim/%s' % name)
        if elem is not None:
            elem.text = " " + str(int(params[name])) + " "

    if dest:
        tree.write(dest, xml_declaration=True, pretty_print=True)
    else:
        tree.write(source, xml_declaration=True, pretty_print=True)

    return


def measureEI(outputpath):
    """
    Measures the Elongation Index of a RBC by fitting an ellipsoid to the RBC mesh
    """

    configpath = "%s/config.xml" % outputpath
    tree = etree.parse(configpath, parser=etree.XMLParser(remove_blank_text=True, remove_comments=True))
    root = tree.getroot()

    tmax = root.find("sim/tmax")
    t = int(float(tmax.text) + 0.5)

    datapath = "%s/tmp/hdf5/" % (outputpath)

    # Return None if the Lisa job crashes
    try:
        fluid, rbc, platelet, ct3 = HCELL_readhdf5.open_hdf5_files(p=False, f=False, ct3=False, half=True, 
                                                                   begin=t, end=t+1, timestep=1, datapath=datapath)
    except (OSError, IOError):
        return None


    # Measure elongation index
    pos = np.array(rbc[0].position)

    if pos.shape[0] > 0:
        X = pos[:, 0]
        Y = pos[:, 1]
        Z = pos[:, 2]
    else:
        return -100, 0.0
    
    A, B, elongation_index = EL.elongation_index(X, Y)

    if np.isnan(elongation_index):
        return -100, 0.0

    return elongation_index, 0.0
    
def measureEI_full(outputpath):
    configpath = "%s/config.xml" % outputpath
    tree = etree.parse(configpath, parser=etree.XMLParser(remove_blank_text=True, remove_comments=True))
    root = tree.getroot()

    tmax_elem = root.find("sim/tmax")
    tmax = int(float(tmax_elem.text) + 0.5)
    
    tmeas_elem = root.find("sim/tmeas")
    tmeas = int(float(tmeas_elem.text) + 0.5)

    datapath = "%s/tmp/hdf5/" % (outputpath)

    # Return None if the Lisa job crashes
    try:

        fluid, rbc, platelet, ct3 = HCELL_readhdf5.open_hdf5_files(p=False, f=False, ct3=False, half=True, 
                                                                   begin=tmeas, end=tmax+1, timestep=tmeas, datapath=datapath)
    except (OSError, IOError):
        return None

    el = []
    err = []

    # Measure elongation index
    for n in range(len(rbc)):
        pos = np.array(rbc[n].position)
        if pos.shape[0] > 0:
            X = pos[:, 0]
            Y = pos[:, 1]
            Z = pos[:, 2]
            
            A, B, elongation_index = EL.elongation_index(X, Y)
            el.append(elongation_index)
            err.append(0.0)
        else:
            el.append(-100)
            err.append(0.0)

    return np.array(el), np.array(err)
