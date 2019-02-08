# System modules
import sys
import os
import time
import shutil
import subprocess

# Numerics modules
import numpy as np

# Local modules for data reading and measurement
import HCELL_readhdf5
import EL

# Import XML parser
from lxml import etree

# Import environment paths
# from lisa_config import *
from local_config import *

sys.path.append("/home/kevin/master_project/IUQ-Project/measure")


class Model:
    """
    Defines a model object which implements the overhead for running a
    black box model and aggregating the output quantities of interest.

    TODO:
    - Implement dynamic process queueing
    - Implement data cleanup after output measurement
    """

    def __init__(self, setup, modelpath, nprocs=1):
        # Path to the model program
        self.modelpath = modelpath

        # Model setup function
        self.setup = setup

        # Number of processes allowed to run concurrently
        self.nprocs = nprocs

        # List processes currently running
        self.procList = []

        # Run FIFO queue
        self.runQueue = []

        return

    def run(self, run):
        """
        Starts a process for a single model run
        """

        # Define base directory
        baseDir = os.getcwd()

        # Make and go into the output directory
        if run.batch:
            os.makedirs("%s/%s" % (run.batch, run.tag), exist_ok=True)
            os.chdir("%s/%s" % (run.batch, run.tag))
        else:
            os.makedirs(run.tag, exist_ok=True)
            os.chdir(run.tag)

        # Set up for the model run
        args = self.setup(run.params)

        # Run the model
        print("Batch: %s, Run: %s running..." % (run.batch, run.tag))
        with open("stdout.log","wb") as outfile:
            p = subprocess.Popen([self.modelpath] + args, stdout=outfile)

        os.chdir(baseDir)

        return p

    def refreshBuffer(self):
        """
        Refreshes the process buffer by removing terminated processes
        """

        for p in self.procList:
            if p.poll() is not None:
                self.procList.remove(p)

        return

    def pushNext(self):
        """
        Runs the next process in the queue
        """

        if len(self.runQueue) > 0:
            run = self.runQueue.pop(0)
            p = self.run(run)
            run.process = p
            self.procList.append(p)

        return

    def pushQueue(self):
        """
        Fills the process buffer with queued runs
        """

        self.refreshBuffer()
        while len(self.procList) < self.nprocs and len(self.runQueue) > 0:
            self.pushNext()

        return

    def flushQueue(self):
        """
        Flushes the process Queue
        Implemented as a busy wait loop
        """

        while len(self.runQueue) > 0:
            self.pushQueue()
            time.sleep(0.2)

        return

    def wait(self):
        """
        Waits until all processes in the process buffer are completed
        """

        # Flush the process buffer
        for p in self.procList:
            p.wait()
        
        self.refreshBuffer()

        return

    def enqueue(self, run):
        """
        Adds a parameter set to the run queue
        """

        self.runQueue.append(run)
        self.pushQueue()

        return

    def enqueueBatch(self, batch):
        """
        Adds a batch of parameter sets to the run queue
        """

        for run in batch:
            self.runQueue.append(run)

        self.pushQueue()

        return


class Run:
    """
    Run data structure containing a tag and a dictionary with parameters
    """
    def __init__(self, tag, params, batch=None):
        self.tag = tag
        self.params = params
        self.batch = batch
        self.process = None

        return


def setupHemocell(params):
    """
    Implements the setup for a run of the Hemocell model
    Returns the list of command line arguments
    """

    # Unpack parameters
    shrate = params["shearrate"]
    kLink = params["kLink"]
    kBend = params["kBend"]
    viscRatio = params["viscosityRatio"]

    templatepath = "%s/measure/templates" % (libpath)

    # Build and copy the necessary files
    buildMaterialXml("%s/RBC_HO_template.xml" % (templatepath), kLink, kBend, viscRatio, dest="RBC_HO.xml")
    buildConfig("%s/config_template.xml" % (templatepath), shrate, dest="config.xml")
    shutil.copyfile("%s/RBC_HO.pos" % (templatepath), "./RBC_HO.pos")

    return ["config.xml"]


def buildMaterialXml(source, kLink, kBend, viscosityRatio, dest=None):
    """
    Builds an XML file for the RBC material model by reading in a RBC XML template and setting
    the relevant parameters: kLink, kBend and the viscosity ratio
    """

    tree = etree.parse(source, parser=etree.XMLParser(remove_blank_text=True, remove_comments=True))
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
        tree.write(dest, xml_declaration=True, pretty_print=True)
    else:
        tree.write(source, xml_declaration=True, pretty_print=True)

    return


def buildConfig(source, rateval, outDir=None, dest=None):
    """
    Builds a config XML file by reading in a config template and setting the shear rate 
    """

    tree = etree.parse(source, parser=etree.XMLParser(remove_blank_text=True, remove_comments=True))
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
            output = etree.SubElement(parameters, 'outputDirectory')
        else:
            output = output[0]

        output.text = outDir

    if dest:
        tree.write(dest, xml_declaration=True, pretty_print=True)
    else:
        tree.write(source, xml_declaration=True, pretty_print=True)

    return


def measureEI(t, datapath):
    """
    Measures the Elongation Index of a RBC by fitting an ellipsoid to the RBC mesh
    """

    fluid, rbc, platelet, ct3 = HCELL_readhdf5.open_hdf5_files(p=False, f=False, ct3=False, half=True, read_procs=read_procs, 
                                                               nprocs=nprocs, begin=t, end=t+1, timestep=1, datapath=datapath)

    # Measure elongation index
    pos = np.array(rbc[0].position)
    X = pos[:, 0]
    Y = pos[:, 1]
    Z = pos[:, 2]
    
    A, B, elongation_index = EL.elongation_index(X, Y)

    return elongation_index


if __name__ == "__main__":
    nprocs = 1
    read_procs = 1

    model = Model(setupHemocell,modelpath)

    params = {"shearrate":25, "kLink":1, "kBend":1, "viscosityRatio":1}

    foo = Run("foo", params)

    params = {"shearrate":50, "kLink":2, "kBend":2, "viscosityRatio":2}

    bar = Run("bar", params)
    
    model.enqueueBatch([foo,bar])
    model.flushQueue()
    model.wait()

    #p = model.run(foo)
    #p.wait()

    #p = model.run(bar)
    #p.wait()

    print("procList", model.procList)
    print("runQueue", model.runQueue)

    datapath = "./foo/tmp/hdf5/"

    print("Elongation Index:", measureEI(4000, datapath))
