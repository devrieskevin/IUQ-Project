import os
import sys

import numpy as np

from scipy.stats import multivariate_normal

import scheduler


def normal_likelihood(output, data, data_err, comp_err, model_err):
    """
    The standard likelihood function used for the TMCMC algorithm
    """

    covariance_matrix = np.diag(data_err**2 + 
                                comp_err**2 + 
                                (model_err * data)**2)
    
    return multivariate_normal.pdf(data,mean=output,cov=covariance_matrix)


def createBatch(setup,modelpath,tag,var_dicts,param_dict):
    """
    Returns a list of Run objects using the same batch name
    """

    dicts = [dict(param_dict) for n in range(len(var_dicts))]
    
    for n in range(len(var_dicts)):
        dicts[n].update(var_dicts[n])

    batch = [scheduler.Run(setup, modelpath, "run_%i" % (n), dicts[n], batch=tag) for n in range(len(var_dicts))]

    return batch

def pollBatch(batch):
    """
    Polls all runs in a batch
    Returns None if any runs has not yet been completed
    Otherwise returns a return code
    """

    codes = [run.process.poll() for run in batch]

    if None in codes:
        return None
    else:
        return np.sum(codes)

def adaptive_p(p_old, likelihoods, COVtol):
    """
    Implements an adaptive scheduler for the TMCMC control parameter
    """

    p = p_old + 0.1
    
    if p > 1:
        return 1

    return p


def full_prior(params, priors):
    """
    Evaluates all priors and calculates the full prior 
    by assuming all parameters are independent
    """

    evaluations = np.array([priors[n](params[n]) for n in range(len(priors))])
    return np.prod(evaluations)


def sample(problem, n_samples, likelihood_function=normal_likelihood, p_scheduler=adaptive_p,
           cov_scale=0.2, COVtol=1.0, nprocs=1, sleep_time=0.2):
    """
    Implements a Transitional Markov Chain Monte Carlo (TMCMC) sampler
    """

    # Get base working directory
    baseDir = os.getcwd()

    # Initialize model run scheduler
    runscheduler = scheduler.ModelScheduler(nprocs=nprocs, sleep_time=sleep_time)

    # Unpack model metadata and functions
    setup = problem["setup"]
    measure = problem["measure"]
    modelpath = problem["modelpath"]

    # Unpack parameter and design variable names
    model_params = problem["model parameters"] 
    error_params = problem["error parameters"]
    design_vars = problem["design variables"]

    # Unpack experimental data and measurement errors
    x = np.array(problem["input data"])
    
    if len(x.shape) < 2:
        x = x[:,None]
    
    y = np.array(problem["output data"])
    y_err = np.array(problem["data errors"])

    # Modelling error per data point
    model_errors = problem["error mapping"]

    # Unpack prior functions and samplers
    model_priors = problem["model priors"]
    error_priors = problem["error priors"]

    # Get prior functions and prior samplers
    prior_functions = [func for (func,sampler) in model_priors] + [func for (func,sampler) in error_priors]
    prior_samplers = [sampler for (func,sampler) in model_priors] + [sampler for (func,sampler) in error_priors]

    # Dictionary with design variables
    var_dicts = [{design_vars[m]:x[n,m] for m in range(x.shape[1])} for n in range(x.shape[0])]

    ### INITIALIZATION ###

    # Generate samples from the prior distribution
    samples = np.empty((n_samples,len(prior_samplers)))
    for i, sampler in enumerate(prior_samplers):
        samples[:, i] = sampler(n_samples)

    stage = 0

    os.makedirs("stage_%i" % stage)
    os.chdir("stage_%i" % stage)

    param_dicts = [{model_params[m]:samples[n,m] for m in range(len(model_params))} for n in range(n_samples)]
    batches = [createBatch(setup, modelpath, "batch_%i" % (n), var_dicts, param_dicts[n]) for n in range(n_samples)]

    # Run model for all parameter sets
    for batch in batches:
        runscheduler.enqueueBatch(batch)

    runscheduler.flushQueue()
    runscheduler.wait()

    # Retrieve quantities of interest from output
    qoi = np.empty((n_samples,y.shape[0]))
    c_err = np.empty((n_samples,y.shape[0]))
    for n,batch in enumerate(batches):
        for m,run in enumerate(batch):
            qoi[n,m], c_err[n,m] = measure("%s/%s/%s/%s" % (baseDir,"stage_0",run.batch,run.tag))

    # Calculate likelihoods
    m_err_dicts = [{error_params[m]:samples[n,len(model_params) + m] for m in range(len(error_params))} for n in range(n_samples)]
    m_err = np.array([[m_err_dicts[n][model_errors[m]] for m in range(len(model_errors))] for n in range(n_samples)])

    likelihoods = np.array([likelihood_function(qoi[n],y,y_err,c_err[n],m_err[n]) for n in range(n_samples)])

    os.chdir(baseDir)
    stage += 1

    p = 0
    while True:
        ### ANALYSIS ###

        # Calculate next control parameter value
        p_old = p
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
        
        # Sample leaders and determine maximum Markov chain lengths
        sample_indices = np.random.choice(n_samples,size=n_samples,p=weights_norm)
        unique_indices, chain_lengths = np.unique(sample_indices,return_counts=True)

        # Initialize leader samples and statistics
        leaders = samples[unique_indices]
        leader_likelihoods = likelihoods[unique_indices]
        leader_priors = np.array([full_prior(leaders[n], prior_functions) for n in range(leaders.shape[0])])
        
        os.makedirs("stage_%i" % stage)
        os.chdir("stage_%i" % stage)

        counter = np.ones(leaders.shape[0])

        # Generate candidates and create batches per chain
        candidates = np.array([np.random.multivariate_normal(leaders[n],covariance_matrix) for n in range(leaders.shape[0])])

        param_dicts = [{model_params[m]:candidates[n,m] for m in range(len(model_params))} for n in range(candidates.shape[0])]
        chains = [createBatch(setup,modelpath,"chain_%i_batch_%i" % (n,counter[n]),var_dicts,param_dicts[n]) for n in range(leaders.shape[0])]

        # Schedule chain runs
        for chain in chains:
            runscheduler.enqueueBatch(chain)        
        
        runscheduler.flushQueue()

        for n in range(len(chains)):
            if pollBatch(chains[n]) is not None:
                # Measure Quantity of interest
                qoi = np.empty(y.shape[0])
                c_err = np.empty(y.shape[0])
                for m,run in enumerate(chains[n]):
                    qoi[m], c_err[m] = measure("%s/%s/%s/%s" % (baseDir,"stage_%i" % stage,run.batch,run.tag))

                # Modelling errors
                m_err_dict = {error_params[m]:candidates[n,len(model_params) + m] for m in range(len(error_params))}
                m_err = np.array([m_err_dict[model_errors[m]] for m in range(len(model_errors))])

                # Calculate candidate statistics
                candidate_likelihood = likelihood_function(qoi,y,y_err,c_err,m_err)
                candidate_prior = full_prior(candidates[n],prior_functions)

                # Acceptance-Rejection step
                ratio = (candidate_likelihood**p * candidate_prior) / (leader_likelihoods[n]**p * leader_priors[n])
                randnum = np.random.uniform(0,1)

                if randnum <= ratio:
                    # Accept
                else:
                    # Reject
       
        
        os.chdir(baseDir)
        stage += 1
        
        if p >= 1:
            break

    return samples
