import os
import shutil
import sys
import time
import signal

import dill

import numpy as np

from scipy.stats import multivariate_normal
from scipy.optimize import minimize,minimize_scalar

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

    queued = np.array([[run.process is None for run in batch] for batch in batches])
    n_queued = np.sum(queued,axis=1)
    return np.any((n_queued > 0) & (n_queued < queued.shape[1]))

def load_state(logfile):
    with open(logfile,"rb") as f:
        sampler = dill.load(f)
    
    return sampler
    
def COV_biased(x):
    mu = np.mean(x)
    #std = np.sqrt(np.sum((x-mu)**2) / (x.shape[0] - 1))
    std = np.sqrt(np.var(x,ddof=1))
    
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

    #p0 = np.array([p_old+1e-6])
    #res = minimize(objective,p0,method="SLSQP",bounds=((p_old,1.1),))
    #p = res.x[0]
    
    res = minimize_scalar(objective,bounds=(p_old,1.1),method="bounded",options={'xatol':1e-12,'maxiter':1000})
    p = res.x

    if p > 1:
        p = 1.0

    print("p:", p)
    print("COV:", COV_biased(likelihoods**(p - p_old)))

    return p

class TMCMC():
    def __init__(self, problem, likelihood_function=normal_likelihood, p_scheduler=adaptive_p,
                 cov_scale=0.2, COVtol=1.0, nburn=0, lmax=np.inf,
                 nprocs=1, sleep_time=0.2, logstep=50, logpath="TMCMC_log.pkl"):

        # Unpack default keyword arguments
        self.likelihood_function = likelihood_function
        self.p_scheduler = p_scheduler
        self.cov_scale = cov_scale
        self.COVtol = COVtol
        self.nburn = nburn
        self.lmax = lmax
        self.logstep = logstep
        self.logpath = logpath

        self.model_type = problem.get("model_type",None)
    
        # Unpack model functions according to model type
        if self.model_type == "external":
            # Initialize model run scheduler
            self.runscheduler = scheduler.ModelScheduler(nprocs=nprocs, sleep_time=sleep_time)

            self.sleep_time = sleep_time
            
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
            if self.model_type == "external_cluster":
                # Store cluster variables to local
                server = self.runscheduler.server
                clients = self.runscheduler.clients

                # Reset cluster variables for logfile
                self.runscheduler.server = None
                self.runscheduler.clients = {}

            # Save the state of the RNG for reproducibility
            self.rng_state = np.random.get_state()

            with open("TMCMC_log.pkl","wb") as f:
                dill.dump(self,f)

            shutil.copy2("TMCMC_log.pkl", self.logpath)

            if self.model_type == "external_cluster":
                # Restore cluster variables from local
                self.runscheduler.server = server
                self.runscheduler.clients = clients

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

        #print("Min weight norm:",self.weights_norm.min())
        #print("Max weight norm:",self.weights_norm.max())
        #print("Mean weight norm:",self.weights_norm.mean())
        #print("std weight norm:",np.sqrt(np.var(self.weights_norm,ddof=1)))
        #print(self.weights_norm)
        
        # Estimate the covariance matrix for MCMC candidate sampling
        samples_mean = np.sum(self.weights_norm[:,None] * self.samples, axis=0)
        samples_centered = self.samples - samples_mean[None,:]
        centered_outer = samples_centered[:,:,None] @ samples_centered[:,None,:]

        # Proposal distribution covariance matrix
        self.cov_matrix = self.cov_scale**2 * np.sum(self.weights_norm[:,None,None] * centered_outer,axis=0)

        return

    def advance_MH(self):
        for n in reversed(range(self.leaders.shape[0])):
            if self.counter[n] < self.chain_lengths[n]:
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
                    self.leader_c_err[n] = c_err
                    self.leader_likelihoods[n] = candidate_likelihood
                    self.leader_priors[n] = candidate_prior

                if self.counter[n] >= 0:
                    # Set new leader as a new sample
                    idx = np.sum(self.chain_lengths[:n]) + self.counter[n]
                    self.samples[idx] = self.leaders[n]
                    self.qoi[idx] = self.leader_qoi[n]
                    self.c_err[idx] = self.leader_c_err[n]
                    self.likelihoods[idx] = self.leader_likelihoods[n]

                # Update chain counters
                self.counter[n] += 1
                self.chain_sum += 1

                # Generate new candidate if chain is not complete
                while self.counter[n] < self.chain_lengths[n]:
                    # Generate new candidate
                    self.candidates[n] = np.random.multivariate_normal(self.leaders[n],self.cov_matrix)

                    if self.full_prior(self.candidates[n]) == 0:
                        # Set leader by default and sample a new candidate

                        if self.counter[n] >= 0:
                            idx = np.sum(self.chain_lengths[:n]) + self.counter[n]
                            self.samples[idx] = self.leaders[n]
                            self.qoi[idx] = self.leader_qoi[n]
                            self.c_err[idx] = self.leader_c_err[n]
                            self.likelihoods[idx] = self.leader_likelihoods[n]

                        # Update chain counters
                        self.counter[n] += 1
                        self.chain_sum += 1

                    else:
                        # Set new candidate in the chain and add to the run queue
                        self.param_dicts[n] = {self.model_params[m]:self.candidates[n,m] for m in range(len(self.model_params))}
                    
                        if self.model_type in ["external","external_cluster"]:
                            path = "stage_%i" % (self.stage)
                            self.chains[n] = createBatch(self.setup,"chain_%i_batch_%i" % 
                                                         (n,self.counter[n]),self.var_dicts,self.param_dicts[n],path,self.measure)
                        
                            self.runscheduler.prependBatch(self.chains[n])

                        break

        return

    def resample(self, n_samples):
        # Initialize resampling step
        if self.counter is None:
            # Sample leaders and determine maximum Markov chain lengths
            sample_indices = np.random.choice(n_samples,size=n_samples,p=self.weights_norm)

            if self.lmax == np.inf or self.lmax == float("inf"):
                sample_params = self.samples[sample_indices]
                unique_params,unique_indices,unique_counts = np.unique(sample_params,return_index=True,
                                                                       return_counts=True,axis=0)

                # Sort chain lengths in descending order
                length_sort = unique_counts.argsort()[::-1]

                self.chain_lengths = unique_counts[length_sort]
                leaders = unique_params[length_sort]
                leader_qoi = self.qoi[sample_indices][unique_indices][length_sort]
                leader_c_err = self.c_err[sample_indices][unique_indices][length_sort]
                leader_likelihoods = self.likelihoods[sample_indices][unique_indices][length_sort]

            elif self.lmax == 1:
                leaders = self.samples[sample_indices]
                leader_qoi = self.qoi[sample_indices]
                leader_c_err = self.c_err[sample_indices]
                leader_likelihoods = self.likelihoods[sample_indices]
                self.chain_lengths = np.ones(n_samples,dtype=int)
            else:
                # TODO
                sample_params = self.samples[sample_indices]
                unique_params,unique_counts = np.unique(sample_params,return_counts=True,axis=0)
                for n in range(unique_params.shape[0]):
                    if unique_counts[n] > self.lmax:
                        quotient = unique_counts[n] // self.lmax
                        remainder = unique_counts[n] % self.lmax

                        counts = np.array([self.lmax for n in range(quotient)] + [remainder])

            # Initialize leader samples and statistics
            self.leaders = leaders
            self.leader_qoi = leader_qoi
            self.leader_c_err = leader_c_err
            self.leader_likelihoods = leader_likelihoods
            self.leader_priors = np.array([self.full_prior(self.leaders[n]) for n in range(self.leaders.shape[0])])

            #print("Min likelihoods:",self.likelihoods[sample_indices].min())
            #print("Max likelihoods:",self.likelihoods[sample_indices].max())
            #print("Mean likelihoods:",self.likelihoods[sample_indices].mean())
            #print("Num higher likelihoods:",np.sum(self.likelihoods[sample_indices] > 1))
            print("Maximum chain length:",self.chain_lengths.max())
            #print("Likelihood max chain:",self.leader_likelihoods[self.chain_lengths.argmax()])
            
            #if self.p == 1:
            #    print("Leader data:")
            #    print(np.column_stack([self.weights_norm[sample_indices][unique_indices][length_sort],
            #                           self.chain_lengths,self.leader_likelihoods]))
            
            # Initialize counter
            self.counter = np.zeros(self.chain_lengths.shape,dtype=int) - self.nburn
            self.chain_sum = -self.chain_lengths.shape[0] * self.nburn

            # Generate candidates and create batches per chain
            self.candidates = np.array([np.random.multivariate_normal(self.leaders[n],self.cov_matrix) 
                                        for n in range(self.leaders.shape[0])])

            self.param_dicts = [{self.model_params[m]:self.candidates[n,m] for m in range(len(self.model_params))} 
                                for n in range(self.candidates.shape[0])]
        
            if self.model_type in ["external","external_cluster"]:
                path = "stage_%i" % (self.stage)
                self.chains = [createBatch(self.setup,"chain_%i_batch_%i" % 
                                           (n,self.counter[n]),self.var_dicts,self.param_dicts[n],path,self.measure) 
                               for n in range(self.leaders.shape[0])]

                # Schedule chain runs
                for chain in self.chains:
                    self.runscheduler.enqueueBatch(chain)
                            
        while self.chain_sum < n_samples:
            self.advance_MH()

            if self.chain_sum >= self.lognext:
                if self.model_type in ["external","external_cluster"]:
                    while self.runscheduler.batchesQueuedAndRunning(self.chains):
                        self.runscheduler.pushNext()
                        time.sleep(self.sleep_time)

                    self.runscheduler.wait()
                    self.advance_MH()

                self.lognext = (self.chain_sum // self.logstep + 1) * self.logstep
                self.save_state()

            if self.model_type in ["external","external_cluster"]:
                self.runscheduler.pushQueue()
                time.sleep(self.sleep_time)

        print("Current max likelihood:",np.max(self.likelihoods))
        
        self.counter = None
        self.lognext = self.logstep
        self.stage += 1

        return

    def sample(self, n_samples=100, checkpoint=False):
        if checkpoint:
            # Set back RNG state
            np.random.set_state(self.rng_state)

            if self.model_type == "external_MPI":
                self.runscheduler.set_MPI_variables()

            if self.model_type == "external_cluster":
                self.runscheduler.reset_server()
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
            
        print("Sampling finished :)")

        if self.model_type == "external_MPI":
            self.runscheduler.finalize()

        if self.model_type == "external_cluster":
            self.runscheduler.close()

        df = pd.DataFrame(data=self.samples,columns=self.model_params+self.error_params)
        df["likelihood"] = self.likelihoods

        return df,self.qoi,self.c_err
