# System modules
import os
import time
import subprocess

# Import environment paths
# from lisa_config import *
from local_config import *


class ModelScheduler:
    """
    Defines a model object which implements the overhead for running a
    black box model and aggregating the output quantities of interest.

    TODO:
    - Implement dynamic process queueing
    - Implement data cleanup after output measurement
    """

    def __init__(self, nprocs=1, sleep_time=0.2):
        # Number of processes allowed to run concurrently
        self.nprocs = nprocs
        
        # Set initial sleep time
        self.sleep_time = sleep_time

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
        args = run.setup(run.path, run.params)

        # Run the model
        print("Batch: %s, Run: %s running..." % (run.batch, run.tag))
        with open("stdout.log","wb") as outfile:
            p = subprocess.Popen(args, stdout=outfile)

        os.chdir(baseDir)

        return p

    def refreshBuffer(self):
        """
        Refreshes the process buffer by removing terminated processes
        """

        self.procList = [p for p in self.procList if p.poll() is None]
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
            time.sleep(self.sleep_time)

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
    def __init__(self, setup, path, tag, params, batch=None):
        self.setup = setup
        self.path = path
        self.tag = tag
        self.params = params
        self.batch = batch
        self.process = None

        return


if __name__ == "__main__":
    import hemocell.model as hemocell

    scheduler = ModelScheduler(nprocs=2)

    params = {"shearrate":25, "kLink":1, "kBend":1, "viscosityRatio":1}

    foo = Run(hemocell.setup, modelpath, "foo", params)

    params = {"shearrate":50, "kLink":2, "kBend":2, "viscosityRatio":2}

    bar = Run(hemocell.setup, modelpath, "bar", params)
    
    scheduler.enqueueBatch([foo,bar])
    scheduler.flushQueue()
    scheduler.wait()

    print("procList", scheduler.procList)
    print("runQueue", scheduler.runQueue)

    bla = Run(hemocell.setup, modelpath, "bla", params)

    scheduler.enqueue(bla)
    scheduler.flushQueue()

    print("procList", scheduler.procList)

    scheduler.wait()

    datapath = "./foo"

    print("Elongation Index:", hemocell.measureEI(4000, datapath))
