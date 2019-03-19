import os
import time
import scheduler

import numpy as np

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

def run_external(problem, samples, nprocs=1, sleep_time=0.2):

    # Get base working directory
    baseDir = os.getcwd()
    
    # Initialize model run scheduler
    runscheduler = scheduler.ModelScheduler(nprocs=nprocs, sleep_time=sleep_time)
    
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

    param_dicts = [{params[m]:samples[n,m] for m in range(len(params))} 
                   for n in range(samples.shape[0])]

    print("Evaluating...")

    # Initialize output buffers
    qoi = np.empty((samples.shape[0],x.shape[0]))
    c_err = np.empty((samples.shape[0],x.shape[0]))
    
    evaluated = np.zeros(samples.shape[0],dtype=bool)
    batches = [createBatch(setup, "batch_%i" % (n), var_dicts, param_dicts[n]) for n in range(samples.shape[0])]

    # Enqueue all sample batches
    for batch in batches:
        runscheduler.enqueueBatch(batch)

    # Run all batches and retrieve quantities of interest
    while not np.all(evaluated):
        for n,batch in enumerate(batches):
            if evaluated[n] or pollBatch(batch) is None:
                continue

            failed = False
            for m,run in enumerate(batch):
                outDir = "%s/%s/%s" % (baseDir,run.batch,run.tag)
                measurement = measure(outDir)
                
                if measurement is not None:
                    qoi[n,m], c_err[n,m] = measurement
                else:
                    print("Simulation in %s failed, requeued" % outDir)
                    runscheduler.requeueRun(run)
                    failed = True

            if not failed:
                evaluated[n] = True

        runscheduler.pushQueue()
        time.sleep(sleep_time)
    
    return qoi, c_err
    
def run_batch(problem, sample, tmax, tmeas, nprocs=1, sleep_time=0.2):

    # Get base working directory
    baseDir = os.getcwd()
    
    # Initialize model run scheduler
    runscheduler = scheduler.ModelScheduler(nprocs=nprocs, sleep_time=sleep_time)
    
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
    
    # Add tmax and tmeas to the parameter dictionary
    param_dict["tmax"] = tmax
    param_dict["tmeas"] = tmeas

    print("Evaluating...")

    # Initialize output buffers
    qoi = np.empty((np.arange(tmeas,tmax+1,tmeas).shape[0],x.shape[0]))
    c_err = np.empty((np.arange(tmeas,tmax+1,tmeas).shape[0],x.shape[0]))
    
    batch = createBatch(setup, "hemocell_batch", var_dicts, param_dict)
    runscheduler.enqueueBatch(batch)

    # Run batch and retrieve quantities of interest
    evaluated = False
    while not evaluated:
        if pollBatch(batch) is not None:
            failed = False
            for m,run in enumerate(batch):
                outDir = "%s/%s/%s" % (baseDir,run.batch,run.tag)
                measurement = measure(outDir)
                
                if measurement is not None:
                    qoi[:,m], c_err[:,m] = measurement
                else:
                    print("Simulation in %s failed, requeued" % outDir)
                    runscheduler.requeueRun(run)
                    failed = True

            if not failed:
                evaluated = True

        runscheduler.pushQueue()
        time.sleep(sleep_time)
    
    return qoi, c_err
