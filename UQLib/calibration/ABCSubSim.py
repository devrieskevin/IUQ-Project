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

def createBatch(setup,tag,var_dicts,param_dict):
    """
    Returns a list of Run objects using the same batch name
    """

    dicts = [dict(param_dict) for n in range(len(var_dicts))]
    
    for n in range(len(var_dicts)):
        dicts[n].update(var_dicts[n])

    batch = [scheduler.Run(setup, "run_%i" % (n), dicts[n], batch=tag) for n in range(len(var_dicts))]

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
            # Get base working directory
            self.baseDir = os.getcwd()
            
            # Initialize model run scheduler
            self.runscheduler = scheduler.ModelScheduler(nprocs=nprocs, sleep_time=sleep_time)
            
            self.setup = problem["setup"]
            self.measure = problem["measure"]
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
            # Save the state of the RNG for reproducibility
            self.rng_state = np.random.get_state()

            with open("ABCSubSim_log.pkl","wb") as f:
                dill.dump(self,f)

            shutil.copy2("ABCSubSim_log.pkl", self.logpath)

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
        
        if self.model_type == "external":
            self.initialized = np.zeros(n_samples,dtype=bool)
        
            os.makedirs("stage_0")
            os.chdir("stage_0")
            
            batches = [createBatch(self.setup, "batch_%i" % (n), self.var_dicts, param_dicts[n]) for n in range(n_samples)]

            # Enqueue all sample batches
            for batch in batches:
                self.runscheduler.enqueueBatch(batch)

            # Run all batches and retrieve quantities of interest
            while not np.all(self.initialized):
                for n,batch in enumerate(batches):
                    if self.initialized[n] or pollBatch(batch) is None:
                        continue

                    failed = False
                    for m,run in enumerate(batch):
                        outDir = "%s/%s/%s/%s" % (self.baseDir,"stage_%i" % self.stage,run.batch,run.tag)
                        measurement = self.measure(outDir)
                        
                        if measurement is not None:
                            self.qoi[n,m], self.c_err[n,m] = measurement
                        else:
                            print("Simulation in %s failed, requeued" % outDir)
                            self.runscheduler.requeueRun(run)
                            failed = True

                    if failed:
                        continue

                    self.initialized[n] = True

                self.runscheduler.pushQueue()
                time.sleep(self.sleep_time)

            os.chdir(self.baseDir)
                    
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
            
                if self.model_type == "external":
                    self.chains[n] = createBatch(self.setup,"chain_%i_batch_%i" % (n,self.counter[n]),
                                                 self.var_dicts,self.param_dicts[n])
                    
                    self.runscheduler.enqueueBatch(self.chains[n])

                break

            else:
                # Reject and set current leader as a new sample
                idx = n * self.invP0 + self.counter[n]
                self.samples[idx] = self.leaders[n]
                self.qoi[idx] = self.leader_qoi[n]
                self.distances[idx] = self.leader_distances[n]

                # Update chain counter
                self.counter[n] += 1
                self.chain_sum += 1

        return
                
    def advance_MMA(self):
        for n in range(self.sample_min,self.sample_max):
            if self.counter[n] < self.invP0:
                if self.model_type == "external" and pollBatch(self.chains[n]) is None:
                    continue
                
                # Measure Quantity of interest
                candidate_qoi = np.empty(self.y.shape[0])
                c_err = np.empty(self.y.shape[0])
                
                if self.model_type == "external":
                    failed = False
                    for m,run in enumerate(self.chains[n]):
                        outDir = "%s/%s/%s/%s" % (self.baseDir,"stage_%i" % self.stage,run.batch,run.tag)
                        measurement = self.measure(outDir)
                        
                        if measurement is not None:
                            candidate_qoi[m], c_err[m] = measurement
                        else:
                            print("Simulation in %s failed, requeued" % outDir)
                            self.runscheduler.requeueRun(run)
                            failed = True

                    if failed:
                        continue
                
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
                    self.leader_distances[n] = candidate_distance
                    
                    self.accepted += 1

                # Set new leader as a new sample
                idx = n * self.invP0 + self.counter[n]
                self.samples[idx] = self.leaders[n]
                self.qoi[idx] = self.leader_qoi[n]
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

            # Compute current tolerance value
            self.eps = (self.distances[n_samples // self.invP0 - 1] + self.distances[n_samples // self.invP0]) / 2
            print("Current tolerance value:", self.eps)

            # Set samples with the smallest distances as leaders
            self.leaders = self.samples[:n_samples // self.invP0,:]
            self.leader_qoi = self.qoi[:n_samples // self.invP0,:]
            self.leader_distances = self.distances[:n_samples // self.invP0]

            # Permute leader samples for self-regulated MMA sampling
            permuted_indices = np.random.permutation(self.leaders.shape[0])
            self.leaders = self.leaders[permuted_indices]
            self.leader_qoi = self.leader_qoi[permuted_indices]
            self.leader_distances = self.leader_distances[permuted_indices]

            # Set first samples of next stage to leaders
            self.samples[::self.invP0] = self.leaders
            self.qoi[::self.invP0] = self.leader_qoi
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
            self.chains = [[] for n in range(self.leaders.shape[0])]

        if self.model_type == "external":
            os.makedirs("stage_%i" % self.stage)
            os.chdir("stage_%i" % self.stage)

        while self.group < self.invPa:
            # Calculate variances, and sample and enqueue initial candidates
            if np.sum(self.counter[self.sample_min:self.sample_max]) == (n_samples // self.invP0 // self.invPa):
                self.variances = self.lamb * np.var(self.leaders[self.sample_min:self.sample_max],ddof=1,axis=0)
                for n in range(self.sample_min, self.sample_max):
                    self.sample_unique_candidate(n)
        
            while not np.all(self.counter[self.sample_min:self.sample_max] == self.invP0):
                self.advance_MMA()

                if self.chain_sum >= self.lognext:
                    if self.model_type == "external":
                        while batchesQueuedAndRunning(self.chains[self.sample_min:self.sample_max]):
                            self.runscheduler.pushNext()
                            time.sleep(self.sleep_time)

                        self.runscheduler.wait()
                        self.advance_MMA()

                    self.lognext = (self.chain_sum // self.logstep + 1) * self.logstep
                    self.save_state()

                if self.model_type == "external":
                    self.runscheduler.pushQueue()
                    time.sleep(self.sleep_time)
   
            # Update variables for next chain group
            self.group += 1
            self.sample_min = self.sample_max
            self.sample_max += n_samples // self.invP0 // self.invPa
                        
            self.alpha = self.accepted / (n_samples // self.invP0 // self.invPa) / (self.invP0 - 1)
            self.update_regulation_variables()
            
            self.accepted = 0
            self.save_state()

            #print("Acceptance rate:", self.alpha)
               
        if self.model_type == "external":
            os.chdir(self.baseDir)

        self.counter = None
        self.lognext = self.logstep
        self.stage += 1

        return

    def sample(self, n_samples, checkpoint=False):
        # Assert that (n_samples / invP0) and invP0 are both integers
        assert (n_samples // self.invP0 * self.invP0) == n_samples
        
        # Assert that (n_samples / invP0 / invPa) and invPa are both integers
        assert (n_samples // self.invP0 // self.invPa * self.invPa) == (n_samples // self.invP0)

        if not checkpoint:
            self.initialize(n_samples)
            self.save_state()
        
        while self.stage <= self.max_stages:
            print("Running stage %i..." % self.stage)

            self.iterate_SubSim(n_samples)
            self.save_state()
            
            if self.eps <= self.tol:
                print("Minimum tolerance value reached")
                break    

        df = pd.DataFrame(data=self.samples,columns=self.model_params)
        df["distance"] = self.distances
        
        return df,self.qoi

def sample(problem, n_samples, invP0=5, tol=0.1, max_stages=20, nprocs=1, sleep_time=0.2):
    """
    Approximate Bayesian Computation Sampler

    DEPRECATED. USE THE ABCSubSim CLASS INSTEAD.
    """

    # Assert that (n_samples / invP0) and invP0 are both integers
    assert (n_samples // invP0 * invP0) == n_samples

    model_type = problem.get("model_type",None)
    
    # Unpack model functions according to model type
    if model_type == "external":
        # Get base working directory
        baseDir = os.getcwd()
        
        # Initialize model run scheduler
        runscheduler = scheduler.ModelScheduler(nprocs=nprocs, sleep_time=sleep_time)
        
        setup = problem["setup"]
        measure = problem["measure"]
    elif model_type == "python":
        evaluate = problem["evaluate"]
    else:
        print("No valid model_type specified in problem.")
        return None

    # Unpack parameter and design variable names
    model_params = problem["model_params"] 
    design_vars = problem["design_vars"]

    # Unpack distance measure
    distance = problem["distance"]

    # Unpack experimental data and measurement errors
    x = np.array(problem["input_data"])
    
    if len(x.shape) < 2:
        x = x[:,None]
    
    y = np.array(problem["output_data"])
    y_err = np.array(problem["data_errors"])

    # Unpack prior functions and samplers
    priors = problem["priors"]
    samplers = problem["samplers"]

    # Dictionary with design variables
    var_dicts = [{design_vars[m]:x[n,m] for m in range(x.shape[1])} for n in range(x.shape[0])]

    # Full prior function
    #full_prior = lambda sample: np.prod([priors[n](sample[n]) for n in range(len(priors))])

    ### INITIALIZATION ###

    # Generate samples from the prior distribution
    samples = np.array([samplers[n](n_samples) for n in range(len(samplers))]).T
    
    for n in range(samples.shape[0]):
        if np.any(samples[n] > 5) or np.any(samples[n] < -5):
            print(samples[n])
    
    print("Initializing...")

    param_dicts = [{model_params[m]:samples[n,m] for m in range(len(model_params))} for n in range(n_samples)]
    qoi = np.empty((n_samples,y.shape[0]))
    c_err = np.empty((n_samples,y.shape[0]))
    
    if model_type == "external":
        os.makedirs("stage_0")
        os.chdir("stage_0")
        
        batches = [createBatch(setup, "batch_%i" % (n), var_dicts, param_dicts[n]) for n in range(n_samples)]

        # Run model for all parameter sets
        for batch in batches:
            runscheduler.enqueueBatch(batch)

        runscheduler.flushQueue()
        runscheduler.wait()

        # Retrieve quantities of interest from output
        for n,batch in enumerate(batches):
            for m,run in enumerate(batch):
                qoi[n,m], c_err[n,m] = measure("%s/%s/%s/%s" % (baseDir,"stage_0",run.batch,run.tag))

        os.chdir(baseDir)
                
    elif model_type == "python":
        for n in range(n_samples):
            params = dict(param_dicts[n])
            for m in range(y.shape[0]):
                params.update(var_dicts[m])
                qoi[n,m], c_err[n,m] = evaluate(params)
                
    distances = np.array([distance(qoi[n],y) for n in range(n_samples)])
    
    for stage in range(1,max_stages+1):
        print("Running stage %i..." % stage)

        # Sort samples
        sort_indices = np.argsort(distances)
        distances = distances[sort_indices]
        samples = samples[sort_indices]
        qoi = qoi[sort_indices]

        # Compute current tolerance value
        eps = (distances[n_samples // invP0] + distances[n_samples // invP0 + 1]) / 2
        print("Current tolerance value:", eps)

        # Compute the variances for the proposal distributions of the candidate components
        variances = 0.01 * np.ones(samples.shape[1])

        # Set samples with the smallest distances as leaders
        leaders = samples[:n_samples // invP0,:]
        leader_qoi = qoi[:n_samples // invP0,:]
        leader_distances = distances[:n_samples // invP0]

        # Set first samples of next stage to leaders
        samples[::invP0] = leaders
        qoi[::invP0] = leader_qoi
        distances[::invP0] = leader_distances

        counter = np.ones(leaders.shape[0],dtype=int)

        if model_type == "external":
            os.makedirs("stage_%i" % stage)
            os.chdir("stage_%i" % stage)

        # Generate candidates and create batches per chain
        candidates = np.empty(leaders.shape)
        param_dicts = [{} for n in range(candidates.shape[0])]
        chains = [[] for n in range(leaders.shape[0])]
        for n in range(leaders.shape[0]):
            # Preemptively accept or reject candidates before evaluating
            while counter[n] < invP0:
                # Generate new candidate
                candidates[n] = generate_candidate(leaders[n],priors,variances)

                if np.any(candidates[n] != leaders[n]):

                    # Set new candidate in the chain and add to the run queue
                    param_dicts[n] = {model_params[m]:candidates[n,m] for m in range(len(model_params))}
                 
                    if model_type == "external":
                        chains[n] = createBatch(setup,"chain_%i_batch_%i" % (n,counter[n]),var_dicts,param_dicts[n])
                        runscheduler.enqueueBatch(chains[n])
                    break

                else:
                    # Set new leader as a new sample
                    idx = n * invP0 + counter[n]
                    samples[idx] = leaders[n]
                    qoi[idx] = leader_qoi[n]
                    distances[idx] = leader_distances[n]

                    # Update chain counter
                    counter[n] += 1
        
        if model_type == "external":
            runscheduler.flushQueue()

        while np.sum(counter) < n_samples:
            for n in range(leaders.shape[0]):
                if counter[n] < invP0:
                    
                    if model_type == "external" and pollBatch(chains[n]) is None:
                        continue
                    
                    # Measure Quantity of interest
                    candidate_qoi = np.empty(y.shape[0])
                    c_err = np.empty(y.shape[0])
                    
                    if model_type == "external":
                        for m,run in enumerate(chains[n]):
                            candidate_qoi[m], c_err[m] = measure("%s/%s/%s/%s" % 
                                                                 (baseDir,"stage_%i" % stage,run.batch,run.tag))
                    
                    elif model_type == "python":
                        params = dict(param_dicts[n])
                        for m in range(y.shape[0]):
                            params.update(var_dicts[m])
                            candidate_qoi[m], c_err[m] = evaluate(params)
                                            
                    # Compute candidate distance
                    candidate_distance = distance(candidate_qoi,y)

                    if candidate_distance < eps:
                        # Accept candidate as new leader
                        leaders[n] = candidates[n]
                        leader_qoi[n] = candidate_qoi
                        leader_distances[n] = candidate_distance

                    # Set new leader as a new sample
                    idx = n * invP0 + counter[n]
                    samples[idx] = leaders[n]
                    qoi[idx] = leader_qoi[n]
                    distances[idx] = leader_distances[n]

                    # Update chain counter
                    counter[n] += 1

                    # Generate new candidate if chain is not complete
                    while counter[n] < invP0:
                        # Generate new candidate
                        candidates[n] = generate_candidate(leaders[n],priors,variances)

                        if np.any(candidates[n] != leaders[n]):

                            # Set new candidate in the chain and add to the run queue
                            param_dicts[n] = {model_params[m]:candidates[n,m] for m in range(len(model_params))}
                        
                            if model_type == "external":
                                chains[n] = createBatch(setup,"chain_%i_batch_%i" % (n,counter[n]),var_dicts,param_dicts[n])
                                runscheduler.enqueueBatch(chains[n])

                            break

                        else:
                            # Set new leader as a new sample
                            idx = n * invP0 + counter[n]
                            samples[idx] = leaders[n]
                            qoi[idx] = leader_qoi[n]
                            distances[idx] = leader_distances[n]

                            # Update chain counter
                            counter[n] += 1

            if model_type == "external":
                if runscheduler.queue:
                    runscheduler.flushQueue()
                else:
                    runscheduler.wait()
               
        if model_type == "external":
            os.chdir(baseDir)
        
        if eps <= tol:
            print("Minimum tolerance value reached")
            break    

    df = pd.DataFrame(data=samples,columns=model_params)
    df["distance"] = distances
    
    return df,qoi
