import os
import time
import tqdm
import scheduler

import numpy as np

def createBatch(setup,tag,var_dicts,param_dict,path,measure):
    """
    Returns a list of Run objects using the same batch name
    """

    dicts = [dict(param_dict) for n in range(len(var_dicts))]
    
    for n in range(len(var_dicts)):
        dicts[n].update(var_dicts[n])

    batch = [scheduler.Run(setup, "run_%i" % (n), dicts[n], path=path, measure=measure, batch=tag) for n in range(len(var_dicts))]

    return batch

def run_external(problem, samples, nprocs=1, sleep_time=0.2, path="run_output",keep_output=True):

    model_type = problem.get("model_type",None)

    if model_type == "external":    
        # Initialize model run scheduler
        runscheduler = scheduler.ModelScheduler(nprocs=nprocs, sleep_time=sleep_time,keep_output=keep_output)
    elif model_type == "external_cluster":
        runscheduler = scheduler.ClusterScheduler(nprocs=nprocs,sleep_time=sleep_time,keep_output=keep_output)
        
        # Initialize the server and listen for connection requests
        runscheduler.server_bind()
        runscheduler.server_listen()
        
    else:
        print("No valid model_type specified in problem.")
        return None
    
    setup = problem["setup"]
    measure = problem["measure"]

    # Unpack parameter and design variable names
    params = problem["params"]
    design_vars = problem["design_vars"]

    # Unpack design variable data
    x = np.array(problem["input_data"])
    
    if len(x.shape) < 2:
        x = x[:,None]

    # Dictionary with design variables
    var_dicts = [{design_vars[m]:x[n,m] for m in range(x.shape[1])} for n in range(x.shape[0])]

    param_dicts = [{params[m]:samples[n,m] for m in range(len(params))} 
                   for n in range(samples.shape[0])]

    print("Evaluating...")

    # Initialize output buffers
    qoi = np.empty((samples.shape[0],x.shape[0]))
    c_err = np.empty((samples.shape[0],x.shape[0]))
    
    evaluated = np.zeros(samples.shape[0],dtype=bool)
    
    batches = [createBatch(setup, "batch_%i" % (n), var_dicts, param_dicts[n], path, measure) for n in range(samples.shape[0])]

    # Enqueue all sample batches
    for batch in batches:
        runscheduler.enqueueBatch(batch)

    # Initialize progress bar
    pbar = tqdm.tqdm(total=len(batches))

    # Run all batches and retrieve quantities of interest
    while not np.all(evaluated):
        for n,batch in enumerate(batches):
            if evaluated[n] or runscheduler.pollBatch(batch) is None:
                continue

            for m,run in enumerate(batch):
                qoi[n,m], c_err[n,m] = run.output

            evaluated[n] = True
            pbar.update(1)

        runscheduler.pushQueue()
        time.sleep(sleep_time)
    
    if model_type == "external_cluster":
        runscheduler.close()

    # Close progress bar    
    pbar.close()

    return qoi, c_err
    
def run_batch(problem, sample, nprocs=1, sleep_time=0.2, path="run_output",keep_output=True):

    # Get base working directory
    baseDir = os.getcwd()
    
    # Initialize model run scheduler
    runscheduler = scheduler.ModelScheduler(nprocs=nprocs, sleep_time=sleep_time,keep_output=keep_output)
    
    setup = problem["setup"]
    measure = problem["measure"]

    # Unpack parameter and design variable names
    params = problem["params"]
    design_vars = problem["design_vars"]

    # Unpack experimental data and measurement errors
    x = np.array(problem["input_data"])
    
    if len(x.shape) < 2:
        x = x[:,None]

    # Dictionary with design variables
    var_dicts = [{design_vars[m]:x[n,m] for m in range(x.shape[1])} for n in range(x.shape[0])]
    param_dict = {params[m]:sample[m] for m in range(len(params))}
    
    print("Evaluating...")

    tmax = var_dicts[0]["tmax"]
    tmeas = var_dicts[0]["tmeas"]

    # Initialize output buffers
    qoi = np.empty((np.arange(tmeas,tmax+1,tmeas).shape[0],x.shape[0]))
    c_err = np.empty((np.arange(tmeas,tmax+1,tmeas).shape[0],x.shape[0]))
    
    batch = createBatch(setup, "hemocell_batch", var_dicts, param_dict, path, measure)
    runscheduler.enqueueBatch(batch)

    # Run batch and retrieve quantities of interest
    evaluated = False
    while not evaluated:
        if runscheduler.pollBatch(batch) is not None:
            for m,run in enumerate(batch):                
                qoi[:,m], c_err[:,m] = run.output

            evaluated = True

        runscheduler.pushQueue()
        time.sleep(sleep_time)
    
    return qoi, c_err
