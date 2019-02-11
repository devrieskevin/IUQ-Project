import numpy as np

from .. import scheduler

def sample(problem, n_samples, p_scheduler=adaptive_p, cov_scale=0.2, COVtol=1.0, nprocs=1):
    """
    Implements a Transitional Markov Chain Monte Carlo (TMCMC) sampler
    """

    # Initialize model run scheduler
    runscheduler = scheduler.ModelScheduler(nprocs=nprocs)

    # Unpack path to the model
    setup = problem["setup"]
    modelpath = problem["modelpath"]

    # Unpack parameter and design variable names
    model_params = problem["model parameters"]
    error_params = problem["error parameters"]
    design_vars = problem["design variables"]

    # Unpack experimental data and measurement errors
    x = problem["input data"]
    y = problem["output data"]
    yerr = problem["data errors"]

    # Unpack prior functions and samplers
    model_priors = problem["model priors"]
    error_priors = problem["error priors"]

    # Generate samples from the prior distribution
    samples = np.empty((n_samples,len(model_params + error_params)))
    for i, (func, sampler) in enumerate(model_priors + error_priors):
        samples[:, i] = sampler(n_samples)

    # Run model for all parameter sets
    param_dicts = [{model_params[m]:samples[n,m] for m in range(len(model_parameters))} for n in range(n_samples)]
    batch = [Run(setup,modelpath,"run_%i" % (n), param_dicts[n], batch="prior_runs") for n in range(n_samples)]

    runscheduler.enqueueBatch(batch)
    runscheduler.flushQueue()
    runscheduler.wait()

    # Retrieve quantities of interest from output
    # Implementation here

    # Calculate likelihoods
    # Implementation here
    likelihoods = np.zeros(n_samples)

    p = 0
    while True:
        # ANALYSIS

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

        # RESAMPLE
        sample_indices = np.random.choice(n_samples,size=n_samples,p=weights_norm)

        if p >= 1:
            break

    return samples


def adaptive_p(p_old, likelihoods, COVtol):
    return p_old + 0.1


if __name__ == "__main__":
    pass
