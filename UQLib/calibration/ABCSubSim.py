import os
import shutil
import sys
import time
import signal

import dill

import numpy as np

from scipy.stats import multivariate_normal
from scipy.optimize import minimize

import pandas as pd

import scheduler

def createBatch(setup,tag,var_dicts,param_dict,path,measure):
    """
    Returns a list of Run objects using the same batch name
    """

    dicts = [dict(param_dict) for n in range(len(var_dicts))]
    
    for n in range(len(var_dicts)):
        dicts[n].update(var_dicts[n])

    batch = [scheduler.Run(setup, "run_%i" % (n), dicts[n], path=path, measure=measure, batch=tag) for n in range(len(var_dicts))]

    return batch

def pollBatch(batch):
    """
    Polls all runs in a batch
    Returns None if any runs has not yet been completed
    Otherwise returns a return code
    """

    codes = [(run.process.poll() if run.process else None) for run in batch]

    if None in codes:
        return None
    else:
        return np.sum(codes)

def batchesQueuedAndRunning(batches):
    """
    Checks if any batches in a list has both queued and running/finished model runs
    """

    # Remove empty batches
    created = [batch for batch in batches if batch is not []]

    queued = np.array([[run.process is None for run in batch] for batch in created])
    n_queued = np.sum(queued,axis=1)
    return np.any((n_queued > 0) & (n_queued < queued.shape[1]))

def load_state(logfile):
    with open(logfile,"rb") as f:
        sampler = dill.load(f)
    
    return sampler

class ABCSubSim:
    def __init__(self, problem, invP0=5, invPa=10, alpha_goal=0.44, xi_init=1.0, lamb_init=0.6, 
                 tol=0.1, max_stages=20, nprocs=1, sleep_time=0.2, logstep=50, logpath="ABCSubSim_log.pkl"):

        # Store default arguments
        self.invP0 = invP0
        self.invPa = invPa
        self.alpha_goal = alpha_goal
        self.xi_init = xi_init
        self.lamb_init = lamb_init
        self.tol = tol
        self.max_stages = max_stages
        self.sleep_time = sleep_time
        self.logstep = logstep
        self.logpath = logpath

        self.model_type = problem.get("model_type",None)
        
        # Unpack model functions according to model type
        if self.model_type == "external":
            # Initialize model run scheduler
            self.runscheduler = scheduler.ModelScheduler(nprocs=nprocs, sleep_time=sleep_time)
            
            self.setup = problem["setup"]
            self.measure = problem["measure"]
        elif self.model_type == "external_cluster":
            # Initialize model run scheduler
            self.runscheduler = scheduler.ClusterScheduler(nprocs=nprocs, sleep_time=sleep_time)

            self.sleep_time = sleep_time
            
            self.setup = problem["setup"]
            self.measure = problem["measure"]

            # Initialize the server and listen for connection requests
            self.runscheduler.server_bind()
            self.runscheduler.server_listen()            
        elif self.model_type == "python":
            self.evaluate = problem["evaluate"]
        else:
            print("No valid model_type specified in problem.")
            return None

        # Unpack parameter and design variable names
        self.model_params = problem["model_params"] 
        self.design_vars = problem["design_vars"]

        # Unpack distance measure
        self.distance = problem["distance"]

        # Unpack experimental data and measurement errors
        self.x = np.array(problem["input_data"])
        
        if len(self.x.shape) < 2:
            self.x = self.x[:,None]
        
        self.y = np.array(problem["output_data"])
        self.y_err = np.array(problem["data_errors"])

        # Unpack prior functions and samplers
        self.priors = problem["priors"]
        self.samplers = problem["samplers"]

        # Dictionary with design variables
        self.var_dicts = [{self.design_vars[m]:self.x[n,m] for m in range(self.x.shape[1])} for n in range(self.x.shape[0])]

        # Full prior function
        #self.full_prior = lambda sample: np.prod([self.priors[n](sample[n]) for n in range(len(self.priors))])

        return

    def save_state(self):
        if self.logpath:
            if self.model_type == "external_cluster":
                # Store cluster variables to local
                server = self.runscheduler.server
                clients = self.runscheduler.clients

                # Reset cluster variables for logfile
                self.runscheduler.server = None
                self.runscheduler.clients = {}
        
            # Save the state of the RNG for reproducibility
            self.rng_state = np.random.get_state()

            with open("ABCSubSim_log.pkl","wb") as f:
                dill.dump(self,f)

            shutil.copy2("ABCSubSim_log.pkl", self.logpath)

            if self.model_type == "external_cluster":
                # Restore cluster variables from local
                self.runscheduler.server = server
                self.runscheduler.clients = clients

        return

    def initialize(self, n_samples):
        # Generate samples from the prior distributions
        self.samples = np.array([self.samplers[n](n_samples) for n in range(len(self.samplers))]).T
        
        self.stage = 0
        print("Initializing...")

        param_dicts = [{self.model_params[m]:self.samples[n,m] for m in range(len(self.model_params))} 
                       for n in range(n_samples)]
        
        self.qoi = np.empty((n_samples,self.y.shape[0]))
        self.c_err = np.empty((n_samples,self.y.shape[0]))
        
        if self.model_type in ["external","external_cluster"]:
            self.initialized = np.zeros(n_samples,dtype=bool)
        
            path = "stage_%i" % (self.stage)
            batches = [createBatch(self.setup, "batch_%i" % (n), self.var_dicts, param_dicts[n], path, self.measure) for n in range(n_samples)]

            # Enqueue all sample batches
            for batch in batches:
                self.runscheduler.enqueueBatch(batch)

            # Run all batches and retrieve quantities of interest
            while not np.all(self.initialized):
                for n,batch in enumerate(batches):
                    if self.initialized[n] or self.runscheduler.pollBatch(batch) is None:
                        continue

                    for m,run in enumerate(batch):
                        self.qoi[n,m], self.c_err[n,m] = run.output

                    self.initialized[n] = True

                self.runscheduler.pushQueue()
                time.sleep(self.sleep_time)
                    
        elif self.model_type == "python":
            for n in range(n_samples):
                params = dict(param_dicts[n])
                for m in range(self.y.shape[0]):
                    params.update(self.var_dicts[m])
                    self.qoi[n,m], self.c_err[n,m] = self.evaluate(params)
                    
        self.distances = np.array([self.distance(self.qoi[n],self.y) for n in range(n_samples)])

        # Initialize chain counter
        self.counter = None
        self.lognext = self.logstep
        self.stage += 1

        # Initialize MMA self-regulation parameters
        self.lamb = self.lamb_init

        return

    def update_regulation_variables(self):
        sigmoid = lambda x: 2 / (1 + np.exp(-100 * x)) - 1

        alpha_diff_prev = self.alpha_diff

        # Calculate and store shared regulation variables
        self.alpha_diff = self.alpha - self.alpha_goal
        self.lamb = np.exp(np.log(self.lamb) + self.xi * self.alpha_diff)
        self.xi = max(0,self.xi + sigmoid(alpha_diff_prev * self.alpha_diff))
        
        return

    def generate_candidate(self, sample):
        candidate = np.random.multivariate_normal(sample,np.diag(self.variances))
        for n in range(sample.shape[0]):
            ratio = self.priors[n](candidate[n]) / self.priors[n](sample[n])
            randnum = np.random.uniform(0.0,1.0)
            if randnum > ratio:
                candidate[n] = sample[n]

        return candidate

    def sample_unique_candidate(self, n):
        while self.counter[n] < self.invP0:
            # Generate new candidate
            self.candidates[n] = self.generate_candidate(self.leaders[n])

            if np.any(self.candidates[n] != self.leaders[n]):

                # Set new candidate in the chain and add to the run queue
                self.param_dicts[n] = {self.model_params[m]:self.candidates[n,m] for m in range(len(self.model_params))}
            
                if self.model_type in ["external","external_cluster"]:
                    path = "stage_%i" % self.stage
                    self.chains[n] = createBatch(self.setup,"chain_%i_batch_%i" % (n,self.counter[n]),
                                                 self.var_dicts,self.param_dicts[n],path,self.measure)
                    
                    self.runscheduler.enqueueBatch(self.chains[n])

                break

            else:
                # Reject and set current leader as a new sample
                idx = n * self.invP0 + self.counter[n]
                self.samples[idx] = self.leaders[n]
                self.qoi[idx] = self.leader_qoi[n]
                self.c_err[idx] = self.leader_c_err[n]
                self.distances[idx] = self.leader_distances[n]

                # Update chain counter
                self.counter[n] += 1
                self.chain_sum += 1

        return
                
    def advance_MMA(self):
        for n in range(self.sample_min,self.sample_max):
            if self.counter[n] < self.invP0:
                if self.model_type in ["external","external_cluster"] and self.runscheduler.pollBatch(self.chains[n]) is None:
                    continue
                
                # Measure Quantity of interest
                candidate_qoi = np.empty(self.y.shape[0])
                c_err = np.empty(self.y.shape[0])
                
                if self.model_type in ["external","external_cluster"]:
                    for m,run in enumerate(self.chains[n]):
                        candidate_qoi[m], c_err[m] = run.output
                
                elif self.model_type == "python":
                    params = dict(self.param_dicts[n])
                    for m in range(self.y.shape[0]):
                        params.update(self.var_dicts[m])
                        candidate_qoi[m], c_err[m] = self.evaluate(params)
                                        
                # Compute candidate distance
                candidate_distance = self.distance(candidate_qoi,self.y)

                if candidate_distance < self.eps:
                    # Accept candidate as new leader
                    self.leaders[n] = self.candidates[n]
                    self.leader_qoi[n] = candidate_qoi
                    self.leader_c_err[n] = c_err
                    self.leader_distances[n] = candidate_distance
                    
                    self.accepted += 1

                # Set new leader as a new sample
                idx = n * self.invP0 + self.counter[n]
                self.samples[idx] = self.leaders[n]
                self.qoi[idx] = self.leader_qoi[n]
                self.c_err[idx] = self.leader_c_err[n]
                self.distances[idx] = self.leader_distances[n]

                # Update chain counter
                self.counter[n] += 1
                self.chain_sum += 1

                # Generate new candidates until a unique candidate is sampled
                self.sample_unique_candidate(n)

        return

    def iterate_SubSim(self, n_samples):
        if self.counter is None:
            # Sort samples
            sort_indices = np.argsort(self.distances)
            self.distances = self.distances[sort_indices]
            self.samples = self.samples[sort_indices]
            self.qoi = self.qoi[sort_indices]
            self.c_err = self.c_err[sort_indices]

            # Compute current tolerance value
            self.eps = (self.distances[n_samples // self.invP0 - 1] + self.distances[n_samples // self.invP0]) / 2
            print("Current tolerance value:", self.eps)

            # Set samples with the smallest distances as leaders
            self.leaders = self.samples[:n_samples // self.invP0,:]
            self.leader_qoi = self.qoi[:n_samples // self.invP0,:]
            self.leader_c_err = self.c_err[:n_samples // self.invP0,:]
            self.leader_distances = self.distances[:n_samples // self.invP0]

            # Permute leader samples for self-regulated MMA sampling
            permuted_indices = np.random.permutation(self.leaders.shape[0])
            self.leaders = self.leaders[permuted_indices]
            self.leader_qoi = self.leader_qoi[permuted_indices]
            self.leader_c_err = self.leader_c_err[permuted_indices]
            self.leader_distances = self.leader_distances[permuted_indices]

            # Sample variance of leader samples
            self.leaders_var = np.var(self.leaders,ddof=1,axis=0)

            # Set first samples of next stage to leaders
            self.samples[::self.invP0] = self.leaders
            self.qoi[::self.invP0] = self.leader_qoi
            self.c_err[::self.invP0] = self.leader_c_err
            self.distances[::self.invP0] = self.leader_distances

            # Initialize chain counters
            self.counter = np.ones(self.leaders.shape[0],dtype=int)
            self.chain_sum = np.sum(self.counter)

            # Initialize Self-regulation MMA variables
            self.alpha_diff = 1.0
            self.xi = self.xi_init
            
            # Initialize first chain group
            self.group = 0
            self.sample_min = 0
            self.sample_max = n_samples // self.invP0 // self.invPa
            self.accepted = 0
            
            # Initialize candidates
            self.candidates = np.empty(self.leaders.shape)
            self.param_dicts = [{} for n in range(self.candidates.shape[0])]
            self.chains = [[] for n in range(self.candidates.shape[0])]

        while self.group < self.invPa:
            # Calculate variances, and sample and enqueue initial candidates
            if np.sum(self.counter[self.sample_min:self.sample_max]) == (n_samples // self.invP0 // self.invPa):
                self.variances = self.lamb * self.leaders_var
                for n in range(self.sample_min, self.sample_max):
                    self.sample_unique_candidate(n)
        
            while not np.all(self.counter[self.sample_min:self.sample_max] == self.invP0):
                self.advance_MMA()

                if self.chain_sum >= self.lognext:
                    if self.model_type in ["external","external_cluster"]:
                        while self.runscheduler.batchesQueuedAndRunning(self.chains[self.sample_min:self.sample_max]):
                            self.runscheduler.pushNext()
                            time.sleep(self.sleep_time)

                        self.runscheduler.wait()
                        self.advance_MMA()

                    self.lognext = (self.chain_sum // self.logstep + 1) * self.logstep
                    self.save_state()

                if self.model_type in ["external","external_cluster"]:
                    self.runscheduler.pushQueue()
                    time.sleep(self.sleep_time)
   
            # Update variables for next chain group
            self.group += 1
            self.sample_min = self.sample_max
            self.sample_max += n_samples // self.invP0 // self.invPa
                        
            self.alpha = self.accepted / (n_samples // self.invP0 // self.invPa) / (self.invP0 - 1)
            self.update_regulation_variables()
            
            print("Percentage accepted from group %i: %f" % (self.group,self.alpha))
            
            self.accepted = 0
            self.save_state()

            #print("Acceptance rate:", self.alpha)

        self.counter = None
        self.lognext = self.logstep
        self.stage += 1

        return

    def sample(self, n_samples, checkpoint=False):
        # Assert that (n_samples / invP0) and invP0 are both integers
        assert (n_samples // self.invP0 * self.invP0) == n_samples
        
        # Assert that (n_samples / invP0 / invPa) and invPa are both integers
        assert (n_samples // self.invP0 // self.invPa * self.invPa) == (n_samples // self.invP0)

        if checkpoint:
            # Set back RNG state
            np.random.set_state(self.rng_state)
        
            if self.model_type == "external_cluster":
                self.runscheduler.reset_server()
        else:
            self.initialize(n_samples)
            self.save_state()
        
        while self.stage <= self.max_stages:
            print("Running stage %i..." % self.stage)

            self.iterate_SubSim(n_samples)
            self.save_state()
            
            if self.eps <= self.tol:
                print("Minimum tolerance value reached")
                break    

        print("Sampling finished :)")

        if self.model_type == "external_cluster":
            self.runscheduler.close()

        df = pd.DataFrame(data=self.samples,columns=self.model_params)
        df["distance"] = self.distances
        
        return df,self.qoi,self.c_err
