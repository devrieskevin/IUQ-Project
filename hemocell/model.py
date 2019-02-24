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

    if dest:
        tree.write(dest, xml_declaration=True, pretty_print=True)
    else:
        tree.write(source, xml_declaration=True, pretty_print=True)

    return


def measureEI(t, outputpath):
    """
    Measures the Elongation Index of a RBC by fitting an ellipsoid to the RBC mesh
    """

    datapath = "%s/tmp/hdf5" % (outputpath)

    fluid, rbc, platelet, ct3 = HCELL_readhdf5.open_hdf5_files(p=False, f=False, ct3=False, half=True, 
                                                               begin=t, end=t+1, timestep=1, datapath=datapath)

    # Measure elongation index
    pos = np.array(rbc[0].position)
    X = pos[:, 0]
    Y = pos[:, 1]
    Z = pos[:, 2]
    
    A, B, elongation_index = EL.elongation_index(X, Y)

    return elongation_index
