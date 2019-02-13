import os
import sys

import numpy as np

from scipy.stats import multivariate_normal

import scheduler


def normal_likelihood(output, data, data_err, comp_err, model_err):
    covariance_matrix = np.diag(data_err**2 + 
                                comp_err**2 + 
                                (model_err * data)**2)
    
    return multivariate_normal.pdf(data.flatten(),mean=output,cov=covariance_matrix)


def createBatch(setup,modelpath,tag,var_dicts,param_dict):
    batch = [scheduler.Run(setup, modelpath, "run_%i" % (n), {**param_dict, **var_dicts[n]}, batch=tag) for n in range(len(var_dicts))]

    return batch


def adaptive_p(p_old, likelihoods, COVtol):
    p = p_old + 0.1
    
    if p > 1:
        return 1

    return p


def sample(problem, n_samples, likelihood_function=normal_likelihood, p_scheduler=adaptive_p, cov_scale=0.2, COVtol=1.0, nprocs=1):
    """
    Implements a Transitional Markov Chain Monte Carlo (TMCMC) sampler
    """

    # Get base working directory
    baseDir = os.getcwd()

    # Initialize model run scheduler
    runscheduler = scheduler.ModelScheduler(nprocs=nprocs)

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

    # Dictionary with design variables
    var_dicts = [{design_vars[m]:x[n,m] for m in range(x.shape[1])} for n in range(x.shape[0])]

    
    ### INITIALIZATION ###

    # Generate samples from the prior distribution
    samples = np.empty((n_samples,len(model_params + error_params)))
    for i, (func, sampler) in enumerate(model_priors + error_priors):
        samples[:, i] = sampler(n_samples)

    os.makedirs("stage_0")
    os.chdir("stage_0")

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

    # TEST INITIALIZATION
    return likelihoods

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
        sample_indices = np.random.choice(n_samples,size=n_samples,p=weights_norm)

        if p >= 1:
            break

    return samples


if __name__ == "__main__":
    pass
