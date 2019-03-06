import os
import sys
import time

import dill

import numpy as np

from scipy.stats import multivariate_normal
from scipy.optimize import minimize

import pandas as pd

import scheduler

def normal_likelihood(output, data, data_err, comp_err, model_err):
    """
    The standard likelihood function used for the TMCMC algorithm
    """

    covariance_matrix = np.diag(data_err**2 + 
                                (comp_err * data)**2 + 
                                (model_err * data)**2)
    
    return multivariate_normal.pdf(data,mean=output,cov=covariance_matrix)


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

    queued = np.array([[run.process is None for run in batch] for batch in batches])
    n_queued = np.sum(queued,axis=1)
    return np.any((n_queued > 0) & (n_queued < queued.shape[1]))

def load_state(logfile):
    with open(logfile,"rb") as f:
        sampler = dill.load(f)
    
    return sampler
    
def COV_biased(x):
    mu = np.mean(x)
    std = np.sqrt(np.sum((x-mu)**2) / (x.shape[0] - 1))
    
    return std / mu
    
    
def adaptive_p(p_old, likelihoods, COVtol):
    """
    Implements an adaptive scheduler for the TMCMC control parameter
    """

    #f = np.log(likelihoods)
    #fmax = np.max(f)
    #weight = lambda p: np.exp((f-fmax)*(p - p_old))
        
    weight = lambda p: likelihoods**(p-p_old)
    norm_weight = lambda p: weight(p) / np.sum(weight(p))
    objective = lambda p: np.abs(COV_biased(norm_weight(p)) - COVtol)

    p0 = np.array([p_old+1e-6])

    res = minimize(objective,p0,method="SLSQP",bounds=((p_old,1.1),))

    p = res.x[0]
    
    if p > 1:
        p = 1.0

    print("p:", p)
    print("COV:", COV_biased(likelihoods**(p - p_old)))

    return p

class TMCMC():
    def __init__(self, problem, likelihood_function=normal_likelihood, p_scheduler=adaptive_p,
                 cov_scale=0.2, COVtol=1.0, nprocs=1, sleep_time=0.2, logstep=50, logpath="TMCMC_log.pkl"):

        # Unpack default keyword arguments
        self.likelihood_function = likelihood_function
        self.p_scheduler = p_scheduler
        self.cov_scale = cov_scale
        self.COVtol = COVtol
        self.logstep = logstep
        self.logpath = logpath

        self.model_type = problem.get("model_type",None)
    
        # Unpack model functions according to model type
        if self.model_type == "external":
            # Get base working directory
            self.baseDir = os.getcwd()
            
            # Initialize model run scheduler
            self.runscheduler = scheduler.ModelScheduler(nprocs=nprocs, sleep_time=sleep_time)
            self.sleep_time = sleep_time
            
            self.setup = problem["setup"]
            self.measure = problem["measure"]
        elif self.model_type == "python":
            self.evaluate = problem["evaluate"]
        else:
            print("No valid model_type specified in problem.")
            return None

        # Unpack parameter and design variable names
        self.model_params = problem["model_params"] 
        self.error_params = problem["error_params"]
        self.design_vars = problem["design_vars"]

        # Unpack experimental data and measurement errors
        self.x = np.array(problem["input_data"])
        
        if len(self.x.shape) < 2:
            self.x = self.x[:,None]
        
        self.y = np.array(problem["output_data"])
        self.y_err = np.array(problem["data_errors"])

        # Modelling error per data point
        self.model_errors = problem["error_mapping"]

        # Unpack prior functions and samplers
        self.model_prior = problem["model_prior"]
        self.model_sampler = problem["model_sampler"]

        self.error_prior = problem["error_prior"]
        self.error_sampler = problem["error_sampler"]

        # Construct full prior function
        self.full_prior = lambda sample: self.model_prior(sample[:len(self.model_params)]) * \
                                         self.error_prior(sample[len(self.model_params):])

        # Dictionary with design variables
        self.var_dicts = [{self.design_vars[m]:self.x[n,m] for m in range(self.x.shape[1])} for n in range(self.x.shape[0])]

    def save_state(self):
        if self.logpath:
            # Save the state of the RNG for reproducibility
            self.rng_state = np.random.get_state()

            with open("temp.pkl","wb") as f:
                dill.dump(self,f)

            os.rename("temp.pkl", self.logpath)

        return

    def initialize(self,n_samples):
        # Generate samples from the prior distribution
        self.samples = np.empty((n_samples,len(self.model_params)+len(self.error_params)))
        self.samples[:,:len(self.model_params)] = self.model_sampler(n_samples)
        self.samples[:,len(self.model_params):] = self.error_sampler(n_samples)

        self.stage = 0
        print("Initializing...")

        param_dicts = [{self.model_params[m]:self.samples[n,m] for m in range(len(self.model_params))} 
                       for n in range(n_samples)]
        
        self.qoi = np.empty((n_samples,self.y.shape[0]))
        self.c_err = np.empty((n_samples,self.y.shape[0]))
        
        if self.model_type == "external":
            os.makedirs("stage_%i" % self.stage)
            os.chdir("stage_%i" % self.stage)
            
            batches = [createBatch(self.setup, "batch_%i" % (n), self.var_dicts, param_dicts[n]) for n in range(n_samples)]

            # Run model for all parameter sets
            for batch in batches:
                self.runscheduler.enqueueBatch(batch)

            self.runscheduler.flushQueue()
            self.runscheduler.wait()

            # Retrieve quantities of interest from output
            for n,batch in enumerate(batches):
                for m,run in enumerate(batch):
                    outDir = "%s/%s/%s/%s" % (self.baseDir,"stage_%i" % self.stage,run.batch,run.tag)
                    self.qoi[n,m], self.c_err[n,m] = self.measure(outDir)

            os.chdir(self.baseDir)
                    
        elif self.model_type == "python":
            for n in range(n_samples):
                params = dict(param_dicts[n])
                for m in range(self.y.shape[0]):
                    params.update(self.var_dicts[m])
                    self.qoi[n,m], self.c_err[n,m] = self.evaluate(params)
                    
        # Calculate likelihoods
        m_err_dicts = [{self.error_params[m]:self.samples[n,len(self.model_params) + m] 
                        for m in range(len(self.error_params))} for n in range(n_samples)]
        
        m_err = np.array([[m_err_dicts[n][self.model_errors[m]] 
                           for m in range(len(self.model_errors))] for n in range(n_samples)])

        self.likelihoods = np.array([self.likelihood_function(self.qoi[n],self.y,self.y_err,self.c_err[n],m_err[n]) 
                                     for n in range(n_samples)])
        
        # Initialize chain counter
        self.counter = None
        self.lognext = self.logstep
        self.stage += 1
        self.p = 0

    def analyze(self):
        # Calculate next control parameter value
        self.p_old = self.p

        print("Calculating p for stage %i..." % self.stage)
        self.p = self.p_scheduler(self.p,self.likelihoods,self.COVtol)
        
        # Calculate plausability weights
        weights = self.likelihoods**(self.p - self.p_old)
        weights_mean = np.mean(weights)
        self.weights_norm = weights / np.sum(weights)
        
        # Estimate the covariance matrix for MCMC candidate sampling
        samples_mean = np.sum(self.weights_norm[:,None] * self.samples, axis=0)
        samples_centered = self.samples - samples_mean[None,:]
        centered_outer = samples_centered[:,:,None] @ samples_centered[:,None,:]

        # Proposal distribution covariance matrix
        self.cov_matrix = self.cov_scale**2 * np.sum(self.weights_norm[:,None,None] * centered_outer,axis=0)

        return

    def advance_MH(self):
        for n in range(self.leaders.shape[0]):
            if self.counter[n] < self.chain_lengths[n]:
                
                if self.model_type == "external" and pollBatch(self.chains[n]) is None:
                    continue
                
                # Measure Quantity of interest
                candidate_qoi = np.empty(self.y.shape[0])
                c_err = np.empty(self.y.shape[0])
                
                if self.model_type == "external":
                    for m,run in enumerate(self.chains[n]):
                        candidate_qoi[m], c_err[m] = self.measure("%s/%s/%s/%s" % 
                                                                  (self.baseDir,"stage_%i" % self.stage,run.batch,run.tag))
                
                elif self.model_type == "python":
                    params = dict(self.param_dicts[n])
                    for m in range(self.y.shape[0]):
                        params.update(self.var_dicts[m])
                        candidate_qoi[m], c_err[m] = self.evaluate(params)
                    
                # Modelling errors
                m_err_dict = {self.error_params[m]:self.candidates[n,len(self.model_params) + m] 
                              for m in range(len(self.error_params))}
                
                m_err = np.array([m_err_dict[self.model_errors[m]] for m in range(len(self.model_errors))])

                # Evaluate candidate pdfs
                candidate_likelihood = self.likelihood_function(candidate_qoi,self.y,self.y_err,c_err,m_err)
                candidate_prior = self.full_prior(self.candidates[n])
                
                # Acceptance-Rejection step
                ratio = (candidate_likelihood**self.p * candidate_prior) / \
                        (self.leader_likelihoods[n]**self.p * self.leader_priors[n])
                    
                randnum = np.random.uniform(0,1)
                if randnum < ratio:
                    # Accept candidate as new leader
                    self.leaders[n] = self.candidates[n]
                    self.leader_qoi[n] = candidate_qoi
                    self.leader_likelihoods[n] = candidate_likelihood
                    self.leader_priors[n] = candidate_prior

                # Set new leader as a new sample
                idx = np.sum(self.chain_lengths[:n]) + self.counter[n]
                self.samples[idx] = self.leaders[n]
                self.qoi[idx] = self.leader_qoi[n]
                self.likelihoods[idx] = self.leader_likelihoods[n]

                # Update chain counters
                self.counter[n] += 1
                self.chain_sum += 1

                # Generate new candidate if chain is not complete
                if self.counter[n] < self.chain_lengths[n]:
                    # Generate new candidate
                    self.candidates[n] = np.random.multivariate_normal(self.leaders[n],self.cov_matrix)

                    # Set new candidate in the chain and add to the run queue
                    self.param_dicts[n] = {self.model_params[m]:self.candidates[n,m] for m in range(len(self.model_params))}
                    
                    if self.model_type == "external":
                        self.chains[n] = createBatch(self.setup,"chain_%i_batch_%i" % 
                                                     (n,self.counter[n]),self.var_dicts,self.param_dicts[n])
                        
                        self.runscheduler.enqueueBatch(self.chains[n])

        return

    def resample(self, n_samples):
        # Initialize resampling step
        if self.counter is None:
            # Sample leaders and determine maximum Markov chain lengths
            sample_indices = np.random.choice(n_samples,size=n_samples,p=self.weights_norm)
            unique_indices, self.chain_lengths = np.unique(sample_indices,return_counts=True)

            # Initialize counter
            self.counter = np.zeros(self.chain_lengths.shape,dtype=int)
            self.chain_sum = 0

            # Initialize leader samples and statistics
            self.leaders = self.samples[unique_indices]
            self.leader_qoi = self.qoi[unique_indices]
            self.leader_likelihoods = self.likelihoods[unique_indices]
            self.leader_priors = np.array([self.full_prior(self.leaders[n]) for n in range(self.leaders.shape[0])])

            # Generate candidates and create batches per chain
            self.candidates = np.array([np.random.multivariate_normal(self.leaders[n],self.cov_matrix) 
                                        for n in range(self.leaders.shape[0])])

            self.param_dicts = [{self.model_params[m]:self.candidates[n,m] for m in range(len(self.model_params))} 
                                for n in range(self.candidates.shape[0])]
        
            if self.model_type == "external":
                self.chains = [createBatch(self.setup,"chain_%i_batch_%i" % 
                                           (n,self.counter[n]),self.var_dicts,self.param_dicts[n]) 
                               for n in range(self.leaders.shape[0])]

                # Schedule chain runs
                for chain in self.chains:
                    self.runscheduler.enqueueBatch(chain)

        if self.model_type == "external":
            os.makedirs("stage_%i" % self.stage)
            os.chdir("stage_%i" % self.stage)
                    
        while self.chain_sum < n_samples:
            self.advance_MH()

            if self.chain_sum >= self.lognext:
                if self.model_type == "external":
                    while batchesQueuedAndRunning(self.chains):
                        self.runscheduler.pushNext()
                        time.sleep(self.sleep_time)

                    self.runscheduler.wait()
                    self.advance_MH()

                self.lognext = (self.chain_sum // self.logstep + 1) * self.logstep
                self.save_state()

            if self.model_type == "external":
                self.runscheduler.pushQueue()
                time.sleep(self.sleep_time)

        print("Current max likelihood:",np.max(self.likelihoods))
        
        if self.model_type == "external":
            os.chdir(self.baseDir)

        self.counter = None
        self.lognext = self.logstep
        self.stage += 1

        return

    def sample(self, n_samples=100, checkpoint=False):
        if checkpoint:
            # Set back RNG state
            np.random.set_state(self.rng_state)
        else:
            self.initialize(n_samples)
            self.save_state()
            print("Current max likelihood:",np.max(self.likelihoods))
        
        while self.p < 1:
            if self.counter is None:
                self.analyze()

            print("Running stage %i..." % self.stage)
            self.resample(n_samples)

            self.save_state()
            
        df = pd.DataFrame(data=self.samples,columns=self.model_params+self.error_params)
        df["likelihood"] = self.likelihoods

        return df,self.qoi


def sample(problem, n_samples, likelihood_function=normal_likelihood, p_scheduler=adaptive_p,
           cov_scale=0.2, COVtol=1.0, nprocs=1, sleep_time=0.2):
    """
    Implements a Transitional Markov Chain Monte Carlo (TMCMC) sampler

    DEPRECATED FUNCTION. USE TMCMC CLASS INSTEAD.
    """

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
    error_params = problem["error_params"]
    design_vars = problem["design_vars"]

    # Unpack experimental data and measurement errors
    x = np.array(problem["input_data"])
    
    if len(x.shape) < 2:
        x = x[:,None]
    
    y = np.array(problem["output_data"])
    y_err = np.array(problem["data_errors"])

    # Modelling error per data point
    model_errors = problem["error_mapping"]

    # Unpack prior functions and samplers
    model_prior = problem["model_prior"]
    model_sampler = problem["model_sampler"]

    error_prior = problem["error_prior"]
    error_sampler = problem["error_sampler"]

    # Construct full prior function
    full_prior = lambda sample: model_prior(sample[:len(model_params)]) * error_prior(sample[len(model_params):])

    # Dictionary with design variables
    var_dicts = [{design_vars[m]:x[n,m] for m in range(x.shape[1])} for n in range(x.shape[0])]

    ### INITIALIZATION ###

    # Generate samples from the prior distribution
    samples = np.empty((n_samples,len(model_params)+len(error_params)))
    samples[:,:len(model_params)] = model_sampler(n_samples)
    samples[:,len(model_params):] = error_sampler(n_samples)

    stage = 0
    print("Running stage %i..." % stage)

    param_dicts = [{model_params[m]:samples[n,m] for m in range(len(model_params))} for n in range(n_samples)]
    qoi = np.empty((n_samples,y.shape[0]))
    c_err = np.empty((n_samples,y.shape[0]))
    
    if model_type == "external":
        os.makedirs("stage_%i" % stage)
        os.chdir("stage_%i" % stage)
        
        batches = [createBatch(setup, "batch_%i" % (n), var_dicts, param_dicts[n]) for n in range(n_samples)]

        # Run model for all parameter sets
        for batch in batches:
            runscheduler.enqueueBatch(batch)

        runscheduler.flushQueue()
        runscheduler.wait()

        # Retrieve quantities of interest from output
        for n,batch in enumerate(batches):
            for m,run in enumerate(batch):
                qoi[n,m], c_err[n,m] = measure("%s/%s/%s/%s" % (baseDir,"stage_%i" % stage,run.batch,run.tag))

        os.chdir(baseDir)
                
    elif model_type == "python":
        for n in range(n_samples):
            params = dict(param_dicts[n])
            for m in range(y.shape[0]):
                params.update(var_dicts[m])
                qoi[n,m], c_err[n,m] = evaluate(params)
                
    # Calculate likelihoods
    m_err_dicts = [{error_params[m]:samples[n,len(model_params) + m] 
                    for m in range(len(error_params))} for n in range(n_samples)]
    
    m_err = np.array([[m_err_dicts[n][model_errors[m]] 
                       for m in range(len(model_errors))] for n in range(n_samples)])

    likelihoods = np.array([likelihood_function(qoi[n],y,y_err,c_err[n],m_err[n]) for n in range(n_samples)])
    print("Current max likelihood:",np.max(likelihoods))
    
    stage += 1

    p = 0
    while p < 1:
        ### ANALYSIS ###

        # Calculate next control parameter value
        p_old = p

        print("Calculating p for stage %i..." % stage)
        p = p_scheduler(p,likelihoods,COVtol)
        
        # Calculate plausability weights
        weights = likelihoods**(p - p_old)
        weights_mean = np.mean(weights)
        weights_norm = weights / np.sum(weights)
        
        # Estimate the covariance matrix for MCMC candidate sampling
        samples_mean = np.sum(weights_norm[:,None] * samples, axis=0)
        samples_centered = samples - samples_mean[None,:]
        centered_outer = samples_centered[:,:,None] @ samples_centered[:,None,:]

        covariance_matrix = cov_scale**2 * np.sum(weights_norm[:,None,None] * centered_outer,axis=0)
        
        ### RESAMPLE ###
        
        print("Running stage %i..." % stage)
        
        # Sample leaders and determine maximum Markov chain lengths
        sample_indices = np.random.choice(n_samples,size=n_samples,p=weights_norm)
        unique_indices, chain_lengths = np.unique(sample_indices,return_counts=True)

        # Initialize leader samples and statistics
        leaders = samples[unique_indices]
        leader_qoi = qoi[unique_indices]
        leader_likelihoods = likelihoods[unique_indices]
        leader_priors = np.array([full_prior(leaders[n]) for n in range(leaders.shape[0])])

        counter = np.zeros(leaders.shape[0],dtype=int)

        # Generate candidates and create batches per chain
        candidates = np.array([np.random.multivariate_normal(leaders[n],covariance_matrix) 
                               for n in range(leaders.shape[0])])

        param_dicts = [{model_params[m]:candidates[n,m] for m in range(len(model_params))} 
                       for n in range(candidates.shape[0])]
        
        if model_type == "external":
            os.makedirs("stage_%i" % stage)
            os.chdir("stage_%i" % stage)
            
            chains = [createBatch(setup,"chain_%i_batch_%i" % (n,counter[n]),var_dicts,param_dicts[n]) 
                      for n in range(leaders.shape[0])]

            # Schedule chain runs
            for chain in chains:
                runscheduler.enqueueBatch(chain)        

            runscheduler.flushQueue()

        while np.sum(counter) < n_samples:
            for n in range(leaders.shape[0]):
                if counter[n] < chain_lengths[n]:
                    
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
                        
                    # Modelling errors
                    m_err_dict = {error_params[m]:candidates[n,len(model_params) + m] for m in range(len(error_params))}
                    m_err = np.array([m_err_dict[model_errors[m]] for m in range(len(model_errors))])

                    # Evaluate candidate pdfs
                    candidate_likelihood = likelihood_function(candidate_qoi,y,y_err,c_err,m_err)
                    candidate_prior = full_prior(candidates[n])
                    
                    # Acceptance-Rejection step
                    ratio = (candidate_likelihood**p * candidate_prior) / (leader_likelihoods[n]**p * leader_priors[n])
                    randnum = np.random.uniform(0,1)
                    if randnum < ratio:
                        # Accept candidate as new leader
                        leaders[n] = candidates[n]
                        leader_qoi[n] = candidate_qoi
                        leader_likelihoods[n] = candidate_likelihood
                        leader_priors[n] = candidate_prior

                    # Set new leader as a new sample
                    idx = np.sum(chain_lengths[:n]) + counter[n]
                    samples[idx] = leaders[n]
                    qoi[idx] = leader_qoi[n]
                    likelihoods[idx] = leader_likelihoods[n]

                    # Update chain counter
                    counter[n] += 1

                    # Generate new candidate if chain is not complete
                    if counter[n] < chain_lengths[n]:
                        # Generate new candidate
                        candidates[n] = np.random.multivariate_normal(leaders[n],covariance_matrix)

                        # Set new candidate in the chain and add to the run queue
                        param_dicts[n] = {model_params[m]:candidates[n,m] for m in range(len(model_params))}
                        
                        if model_type == "external":
                            chains[n] = createBatch(setup,"chain_%i_batch_%i" % (n,counter[n]),var_dicts,param_dicts[n])
                            runscheduler.enqueueBatch(chains[n])

            if model_type == "external":
                if runscheduler.queue:
                    runscheduler.flushQueue()
                else:
                    runscheduler.wait()
       
        print("Current max likelihood:",np.max(likelihoods))
        
        if model_type == "external":
            os.chdir(baseDir)
            
        stage += 1

    df = pd.DataFrame(data=samples,columns=model_params+error_params)
    df["likelihood"] = likelihoods

    
    return df,qoi
