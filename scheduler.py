# System modules
import os
import time
import subprocess
import select
import socket

import shutil

import dill

import numpy as np

class ModelScheduler:
    """
    Defines a model object which implements the overhead for running a
    black box model.
    """

    def __init__(self, nprocs=1, sleep_time=0.2, keep_output=True):
        # Number of processes allowed to run concurrently
        self.nprocs = nprocs
        
        # Set initial sleep time
        self.sleep_time = sleep_time

        # Set output flag
        self.keep_output = keep_output

        # List processes currently running
        self.running = [None for n in range(self.nprocs)]

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
            os.makedirs("%s/%s/%s" % (run.path,run.batch,run.tag), exist_ok=True)
            os.chdir("%s/%s/%s" % (run.path,run.batch,run.tag))
        else:
            os.makedirs("%s/%s" % (run.path,run.tag), exist_ok=True)
            os.chdir("%s/%s" % (run.path,run.tag))

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

        for n in range(self.nprocs):
            if self.running[n] is None:
                continue

            if self.running[n].process.poll() is not None:
                if self.running[n].batch:
                    outpath = "%s/%s/%s" % (self.running[n].path,self.running[n].batch,self.running[n].tag)
                else:
                    outpath = "%s/%s" % (self.running[n].path,self.running[n].tag)

                self.running[n].output = self.running[n].measure(outpath,self.running[n].params)

                if not self.keep_output:
                    shutil.rmtree(outpath)

                if self.running[n].output is None:
                    print("Return code:",self.running[n].process.returncode)
                    print("Task failed in %s, restarting task" % outpath)

                    if self.keep_output:
                        shutil.rmtree(outpath)

                    p = self.run(self.running[n])
                    self.running[n].process = p
                else:
                    self.running[n] = None 

        return

    def pushNext(self):
        """
        Runs the next process in the queue
        """

        self.refreshBuffer()
        if len(self.queue) > 0:
            for n in range(self.nprocs):
                if self.running[n] is None:
                    run = self.queue.pop(0)

                    p = self.run(run)
                    
                    run.process = p
                    run.running = True
                    self.running[n] = run

                    break

        return

    def pushQueue(self):
        """
        Fills the process buffer with queued runs
        """

        self.refreshBuffer()
        for n in range(self.nprocs):
            if len(self.queue) == 0:
                break

            if self.running[n] is None:
                run = self.queue.pop(0)

                p = self.run(run)
                    
                run.process = p
                run.running = True
                self.running[n] = run

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

        for n in range(self.nprocs):
            if self.running[n] is None:
                continue

            self.running[n].process.wait()

            if self.running[n].batch:
                outpath = "%s/%s/%s" % (self.running[n].path,self.running[n].batch,self.running[n].tag)
            else:
                outpath = "%s/%s" % (self.running[n].path,self.running[n].tag)

            self.running[n].output = self.running[n].measure(outpath,self.running[n].params)
            while self.running[n].output is None:
                print("Task failed in %s, restarting task" % outpath)
                shutil.rmtree(outpath)
                p = self.run(self.running[n])
                self.running[n].process = p
                self.running[n].process.wait()
                self.running[n].output = self.running[n].measure(outpath,self.running[n].params)

            if not self.keep_output:
                shutil.rmtree(outpath)

            self.running[n] = None

        return

    def requeueRun(self,run):
        """
        Cleans up run output directory and reinserts run into the run queue
        """

        # Remove Run output directory
        if run.batch:
            runDir = "%s/%s/%s" % (run.path,run.batch,run.tag)
        else:
            runDir = "%s/%s" % (run.path,run.tag)

        shutil.rmtree(runDir)

        # Reset Run process to None
        run.process = None

        self.prependRun(run)
        return

    def enqueue(self,run):
        """
        Adds a Run to the run queue
        """

        self.queue.append(run)
        return

    def prependRun(self,run):
        """
        Prepends a Run to the run queue
        """
        
        self.queue = [run] + self.queue
        return

    def prependBatch(self,batch):
        """
        Prepends a batch to the run queue
        """

        self.queue = batch + self.queue
        return

    def enqueueBatch(self,batch):
        """
        Adds a batch of Runs to the run queue
        """

        self.queue = self.queue + batch
        return

    def pollBatch(self,batch):
        """
        Polls all runs in a batch
        Returns None if any of the runs has not yet been completed
        Otherwise returns 0
        """

        outputs = [run.output for run in batch]

        if None in outputs:
            return None
        else:
            return 0

    def batchesQueuedAndRunning(self,batches):
        """
        Checks if any batch in a list has both queued and running/finished model runs
        """

        running = np.array([[run.running for run in batch] for batch in batches],dtype=bool)
        return np.any(np.any(running,axis=1) & ~np.all(running,axis=1))

class MPI_Scheduler:
    """
    Defines a scheduler which implements the overhead for running a
    black box model using MPI.
    """

    def __init__(self, sleep_time=0.2):
        self.set_MPI_variables()

        # Set initial sleep time
        self.sleep_time = sleep_time

        # Run FIFO queue
        self.queue = []

        return

    def set_MPI_variables(self):
        # Set MPI communication variables
        self.comm = MPI.COMM_WORLD
        self.n_workers = self.comm.Get_size()

        # List of MPI workers including self
        self.running = [None for n in range(self.n_workers)]

        return

    def run(self, run):
        """
        Starts a process for a single model run
        """

        # Define base directory
        baseDir = os.getcwd()

        # Make and go into the output directory
        if run.batch:
            os.makedirs("%s/%s/%s" % (run.path,run.batch, run.tag), exist_ok=True)
            os.chdir("%s/%s/%s" % (run.path,run.batch,run.tag))
        else:
            os.makedirs("%s/%s" % (run.path,run.tag), exist_ok=True)
            os.chdir("%s/%s" (run.path,run.tag))

        # Set up for the model run
        args = run.setup(run.params)

        print(args)

        # Run the model
        # print("Batch: %s, Run: %s running..." % (run.batch, run.tag))
        with open("stdout.log","wb") as outfile, open("stderr.log","wb") as errfile:
            p = subprocess.Popen(args, stdout=outfile, stderr=errfile)

        os.chdir(baseDir)

        print("Submitted process")

        return p

    def send_run(self, workerNum, run):
        """
        Send a run to a specified worker
        """

        self.comm.send(run, dest=workerNum)
        req = self.comm.irecv(source=workerNum)
        return req

    def refreshBuffer(self):
        """
        Refreshes the process buffer by removing terminated processes
        """

        for n in range(self.n_workers):
            if self.running[n] is None:
                continue

            if n == 0:
                if self.running[n].process.poll() is not None:
                    if self.running[n].batch:
                        outpath = "%s/%s/%s" % (self.running[n].path,self.running[n].batch,self.running[n].tag)
                    else:
                        outpath = "%s/%s" % (self.running[n].path,self.running[n].tag)

                    self.running[n].output = self.running[n].measure(outpath,self.running[n].params)

                    if self.running[n].output is None:
                        print("Return code:",self.running[n].process.returncode)
                        print("Task failed in worker %i, restarting task" % n)
                        shutil.rmtree(outpath)
                        p = self.run(self.running[n])
                        self.running[n].process = p
                    else:
                        self.running[n] = None            
            else:
                check,value = self.running[n].process.test()

                if check:
                    self.running[n].output = value
                    self.running[n] = None

        return

    def pushNext(self):
        """
        Runs the next process in the queue
        """

        self.refreshBuffer()
        if len(self.queue) > 0:
            for n in range(self.n_workers):
                if self.running[n] is None:
                    run = self.queue.pop(0)

                    if n == 0:
                        p = self.run(run)
                    else:
                        p = self.send_run(n,run)
                    
                    run.process = p
                    self.running[n] = run

                    break

        return

    def pushQueue(self):
        """
        Fills the process buffer with queued runs
        """

        self.refreshBuffer()
        for n in range(self.n_workers):
            if len(self.queue) == 0:
                break

            if self.running[n] is None:
                run = self.queue.pop(0)

                if n == 0:
                    p = self.run(run)
                else:
                    p = self.send_run(n,run)
                    
                run.process = p
                self.running[n] = run

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

        for n in range(self.n_workers):
            if self.running[n] is None:
                continue

            if n == 0:
                self.running[n].process.wait()

                if self.running[n].batch:
                    outpath = "%s/%s/%s" % (self.running[n].path,self.running[n].batch,self.running[n].tag)
                else:
                    outpath = "%s/%s" % (self.running[n].path,self.running[n].tag)

                self.running[n].output = self.running[n].measure(outpath,self.running[n].params)
                while self.running[n].output is None:
                    print("Task failed in worker %i, restarting task" % n)
                    shutil.rmtree(outpath)
                    p = self.run(self.running[n])
                    self.running[n].process = p
                    self.running[n].process.wait()

                self.running[n] = None
            else:
                value = self.running[n].process.wait()
                self.running[n].output = value
                self.running[n] = None

        return
        
    def requeueRun(self, run):
        """
        Cleans up run output directory and reinserts run into the run queue
        """

        # Remove Run output directory
        if run.batch:
            runDir = "%s/%s/%s" % (run.path,run.batch,run.tag)
        else:
            runDir = "%s/%s" % (run.path,run.tag)
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

    def pollBatch(self,batch):
        """
        Polls all runs in a batch
        Returns None if any runs has not yet been completed
        Otherwise returns 0
        """

        outputs = [run.output for run in batch]

        if None in outputs:
            return None
        else:
            return 0

    def batchesQueuedAndRunning(self,batches):
        """
        Checks if any batches in a list has both queued and running/finished model runs
        """

        queued = np.array([[run.process is None for run in batch] for batch in batches])
        n_queued = np.sum(queued,axis=1)
        return np.any((n_queued > 0) & (n_queued < queued.shape[1]))

    def finalize():
        for n in range(self.n_workers):
            self.comm.send(None, dest=n)

        return

class ClusterScheduler:
    """
    Defines a model object which implements the overhead for running a
    black box model using a server-client model to run the model on
    multiple nodes in a computer cluster. The scheduler is implemented
    using sockets. The scheduler requires a client program to be run
    on the corresponding allocated nodes.
    """

    def __init__(self, nprocs=1, sleep_time=0.2, keep_output=True):
        # The server socket used to listen for clients
        self.server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        
        # A dictionary containing the worker clients.
        self.clients = {}

        # Set initial sleep time
        self.sleep_time = sleep_time

        # Number of possible concurrent local processes
        self.scheduler = ModelScheduler(nprocs,sleep_time,keep_output)

        # Run FIFO queue
        self.queue = []

        return

    def reset_server(self):
        # Create a new server socket
        self.server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

        # Reset accepted client dictionary
        self.clients = {}

        # Setup the server again with the previously used parameters
        self.server_bind(self.host,self.port)
        self.server_listen(self.backlog)

        return

    def server_bind(self,host='',port=6677):
        self.host = host
        self.port = port

        self.server.bind((host,port))
        return

    def server_listen(self,backlog=5):
        self.backlog = backlog

        self.server.listen(backlog)
        return

    def accept_client(self):
        client,addr = self.server.accept()

        text = b""
        while not text.endswith(b"\r\n\r\n\r\n"):
            rlist,_,_ = select.select([client],[],[])
            text += client.recv(1024)

        message = text[:-len(b"\r\n\r\n\r\n")]
        client_data = dill.loads(message)

        # Initialize server-side client data
        self.clients[client] = client_data
        self.clients[client]["running"] = [None for n in range(self.clients[client]["nprocs"])]

        print("Server connected to client with %i processes" % self.clients[client]["nprocs"])

        return

    def accept_all_clients(self):
        while True:
            rlist,_,_ = select.select([self.server],[],[],0)

            if rlist:
                self.accept_client()
            else:
                break

        return

    def refreshBuffer(self):
        """
        Refreshes the process buffer by removing 
        finished runs and collecting the model output
        """

        #print("Server command: refreshBuffer")

        # Refresh local buffer
        self.scheduler.refreshBuffer()

        # Get clients with running processes
        refreshable = [client for client in self.clients if np.any([run is not None for run in self.clients[client]["running"]])]

        # Send refresh command to clients
        for client in refreshable:
            data = {"command":"refreshBuffer"}
            message = dill.dumps(data)
            message += b"\r\n\r\n\r\n"
            _,wlist,_ = select.select([],[client],[])
            client.sendall(message)

        # Collect outputs and refresh client buffers
        for client in refreshable:
            text = b""

            while not text.endswith(b"\r\n\r\n\r\n"):
                rlist,_,_ = select.select([client],[],[])
                text += client.recv(1024)

            message = text[:-len(b"\r\n\r\n\r\n")]
            data = dill.loads(message)

            for n,output in data:
                self.clients[client]["running"][n].output = output
                self.clients[client]["running"][n] = None

        return

    def pushNext(self):
        """
        Runs the next process in the queue
        """

        #print("Server command: pushNext")
        #print("Queue length:", len(self.queue))

        self.refreshBuffer()
        self.accept_all_clients()

        if len(self.queue) > 0:
            if None in self.scheduler.running:
                run = self.queue.pop(0)
                self.scheduler.enqueue(run)
                self.scheduler.pushQueue()
            else:
                for client in self.clients:
                    for n in range(self.clients[client]["nprocs"]):
                        if self.clients[client]["running"][n] is None:
                            run = self.queue.pop(0)

                            data = {"command":"run","runs":[(n,run)]}
                            message = dill.dumps(data)
                            message += b"\r\n\r\n\r\n"

                            _,wlist,_ = select.select([],[client],[])
                            client.sendall(message)

                            run.running = True
                            self.clients[client]["running"][n] = run

                            return

        return

    def pushQueue(self):
        """
        Fills the process buffer with queued runs
        """

        #print("Server command: pushQueue")

        self.refreshBuffer()
        self.accept_all_clients()

        # Push local scheduler
        for running in self.scheduler.running:
            if len(self.queue) == 0:
                self.scheduler.pushQueue()
                return

            if running is None:
                run = self.queue.pop(0)
                self.scheduler.enqueue(run)

        self.scheduler.pushQueue()

        for client in self.clients:
            for n in range(self.clients[client]["nprocs"]):
                if len(self.queue) == 0:
                    return

                if self.clients[client]["running"][n] is None:
                    run = self.queue.pop(0)

                    data = {"command":"run","runs":[(n,run)]}
                    message = dill.dumps(data)
                    message += b"\r\n\r\n\r\n"

                    #print("Server message length:",len(message))

                    _,wlist,_ = select.select([],[client],[])
                    client.sendall(message)

                    run.running = True
                    self.clients[client]["running"][n] = run

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

        #print("Server command: wait")

        for client in self.clients:
            data = {"command":"wait"}
            message = dill.dumps(data)
            message += b"\r\n\r\n\r\n"

            _,wlist,_ = select.select([],[client],[])
            client.sendall(message)
        
        self.scheduler.wait()
        self.refreshBuffer()

        return

    def requeueRun(self, run):
        """
        Cleans up run output directory and reinserts run into the run queue
        """

        #TODO

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

    def prependBatch(self,batch):
        """
        Prepends a batch to the run queue
        """

        self.queue = batch + self.queue
        return

    def enqueueBatch(self,batch):
        """
        Adds a batch of Runs to the run queue
        """

        self.queue = self.queue + batch
        return

    def pollBatch(self,batch):
        """
        Polls all runs in a batch
        Returns None if any runs has not yet been completed
        Otherwise returns a return code
        """

        outputs = [run.output for run in batch]

        if None in outputs:
            return None
        else:
            return 0

    def batchesQueuedAndRunning(self,batches):
        """
        Checks if any batches in a list has both queued and running/finished model runs
        """

        running = np.array([[run.running for run in batch] for batch in batches],dtype=bool)
        return np.any(np.any(running,axis=1) & ~np.all(running,axis=1))

    def close(self):
        for client in self.clients:
            data = {"command":"close"}
            message = dill.dumps(data)
            message += b"\r\n\r\n\r\n"

            _,wlist,_ = select.select([],[client],[])
            client.sendall(message)

        self.server.close()
        return

class Run:
    """
    Run data structure containing a tag and a dictionary with parameters
    """
    def __init__(self, setup, tag, params, path=".", measure=None, batch=None):
        self.setup = setup
        self.tag = tag
        self.params = params
        self.batch = batch

        self.path = path
        self.measure = measure

        self.process = None
        self.output = None
        self.running = False

        return

def worker_run(run):
    # Define base directory
    baseDir = os.getcwd()

    # Make and go into the output directory
    if run.batch:
        os.makedirs("%s/%s/%s" % (run.path,run.batch, run.tag), exist_ok=True)
        os.chdir("%s/%s/%s" % (run.path,run.batch,run.tag))
    else:
        os.makedirs("%s/%s" % (run.path,run.tag), exist_ok=True)
        os.chdir("%s/%s" (run.path,run.tag))

    # Set up for the model run
    args = run.setup(run.params)

    # Run the model
    # print("Batch: %s, Run: %s running..." % (run.batch, run.tag))
    with open("stdout.log","wb") as outfile, open("stderr.log","wb") as errfile:
        p = subprocess.Popen(args, stdout=outfile, stderr=errfile)

    os.chdir(baseDir)

    return p

def worker_client():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    baseDir = os.getcwd()

    os.makedirs("Worker_%i" % rank)
    os.chdir("Worker_%i" % rank)

    print("Starting worker %i" % rank)

    while True:
        run = comm.recv(source=0)

        if run is None:
            break

        if run.batch:
            outpath = "%s/%s/%s" % (run.path,run.batch,run.tag)
        else:
            outpath = "%s/%s" % (run.path,run.tag)

        output = None
        while output is None:
            p = worker_run(run)
            p.wait()

            output = run.measure(outpath)

            if output is None:
                #print("cwd:",os.getcwd())
                print("Return code:",p.returncode)
                #with open("%s/stderr.log" % (outpath),'r') as errfile:
                #    print(errfile.read())
                print("Task in worker %i failed, restarting task" % rank)

        comm.send(output,dest=0)

    print("Closing worker %i" % rank)
    os.chdir(baseDir)

    return

if __name__ == "__main__":
    from local_config import *
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
