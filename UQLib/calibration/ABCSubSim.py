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

def generate_candidate(sample,priors,variances):   
    candidate = np.random.multivariate_normal(sample,np.diag(variances))
    for n in range(sample.shape[0]):
        ratio = priors[n](candidate[n]) / priors[n](sample[n])
        randnum = np.random.uniform(0.0,1.0)
        if randnum > ratio:
            candidate[n] = sample[n]

    return candidate

def sample(problem, n_samples, invP0=5, tol=0.1, max_stages=20, nprocs=1, sleep_time=0.2):
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
