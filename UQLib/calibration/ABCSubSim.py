import os
import sys

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

    codes = [run.process.poll() for run in batch]

    if None in codes:
        return None
    else:
        return np.sum(codes)

def sample(problem, n_samples, invP0=5, max_levels=10, nprocs=1, sleep_time=0.2):
    """
    Approximate Bayesian Computation Sampler
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
    model_prior = problem["model_prior"]
    model_sampler = problem["model_sampler"]

    # Dictionary with design variables
    var_dicts = [{design_vars[m]:x[n,m] for m in range(x.shape[1])} for n in range(x.shape[0])]

    ### INITIALIZATION ###

    # Generate samples from the prior distribution
    samples = model_sampler(n_samples)

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
        print("Running level %i..." % stage)

        # Sort samples
        sort_indices = np.argsort(distances)
        distances = distances[sort_indices]
        samples = samples[sort_indices]
        qoi = qoi[sort_indices]

        # Set tolerance value
        eps = 0.5 * (distances[n_samples // invP0] + distances[n_samples // invP0 + 1])

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
