# System modules
import os
import time
import subprocess

import shutil

# Import environment paths
from lisa_config import *
#from local_config import *


class ModelScheduler:
    """
    Defines a model object which implements the overhead for running a
    black box model and aggregating the output quantities of interest.
    """

    def __init__(self, nprocs=1, sleep_time=0.2):
        # Number of processes allowed to run concurrently
        self.nprocs = nprocs
        
        # Set initial sleep time
        self.sleep_time = sleep_time

        # List processes currently running
        self.running = []

        # Run FIFO queue
        self.queue = []

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
        args = run.setup(run.params)

        # Run the model
        # print("Batch: %s, Run: %s running..." % (run.batch, run.tag))
        with open("stdout.log","wb") as outfile, open("stderr.log","wb") as errfile:
            p = subprocess.Popen(args, stdout=outfile, stderr=errfile)

        os.chdir(baseDir)

        return p

    def refreshBuffer(self):
        """
        Refreshes the process buffer by removing terminated processes
        """

        self.running = [p for p in self.running if p.poll() is None]
        return

    def pushNext(self):
        """
        Runs the next process in the queue
        """

        self.refreshBuffer()
        if len(self.running) < self.nprocs and len(self.queue) > 0:
            run = self.queue.pop(0)
            p = self.run(run)
            run.process = p
            self.running.append(p)

        return

    def pushQueue(self):
        """
        Fills the process buffer with queued runs
        """

        self.refreshBuffer()
        while len(self.running) < self.nprocs and len(self.queue) > 0:
            self.pushNext()

        return

    def flushQueue(self):
        """
        Flushes the process Queue
        Implemented as a busy wait loop
        """

        while len(self.queue) > 0:
            self.pushQueue()
            time.sleep(self.sleep_time)

        return

    def wait(self):
        """
        Waits until all processes in the process buffer are completed
        """

        # Flush the process buffer
        for p in self.running:
            p.wait()
        
        self.refreshBuffer()

        return

    def requeueRun(self, run):
        """
        Cleans up run output directory and reinserts run into the run queue
        """

        # Remove Run output directory
        baseDir = os.getcwd()
        if run.batch:
            runDir = "%s/%s/%s" % (baseDir,run.batch,run.tag)
        else:
            runDir = "%s/%s" % (baseDir,run.tag)

        shutil.rmtree(runDir)

        # Reset Run process to None
        run.process = None

        self.prependRun(run)
        return

    def enqueue(self, run):
        """
        Adds a Run to the run queue
        """

        self.queue.append(run)
        return

    def prependRun(self, run):
        """
        Prepends a Run to the run queue
        """
        
        self.queue = [run] + self.queue
        return

    def enqueueBatch(self, batch):
        """
        Adds a batch of Runs to the run queue
        """

        self.queue = self.queue + batch
        return

class Run:
    """
    Run data structure containing a tag and a dictionary with parameters
    """
    def __init__(self, setup, tag, params, batch=None):
        self.setup = setup
        self.tag = tag
        self.params = params
        self.batch = batch
        self.process = None

        return


if __name__ == "__main__":
    import hemocell.model as hemocell

    setup = lambda params: hemocell.setup(modelpath,params)

    scheduler = ModelScheduler(nprocs=2)

    params = {"shearrate":11707.3170732, "kLink":111.69961457, "kBend":121.72321948, "viscosityRatio":11.0079201145}

    foo = Run(setup, "foo", params)
    
    scheduler.enqueueBatch([foo])
    scheduler.flushQueue()

    print("running", scheduler.running)
    print("queue", scheduler.queue)

    scheduler.wait()

    datapath = "./foo"

    print("Elongation Index:", hemocell.measureEI(4000, datapath))
