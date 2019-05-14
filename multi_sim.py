import random
import numpy as np
import pandas as pd

# Require Dask for running independent simulations in parallel
from dask import compute, delayed
import dask.multiprocessing
from dask.diagnostics import ProgressBar
ProgressBar().register()

# Require model for multiple simulations
from gillespie import mRNADynamicsModel as M 

@delayed  # Used to delay execution until scheduled on thread by Dask
def gather_stats(model, tau1, alpha, lambd, N=100_000_000):
    """Gather statistics from Gillespie with different model parameters.
    
    Args: 
        model: Instance of mRNADynamicsModel class
        tau1 (float): transcription factor lifetime
        alpha (float): transcription factor birth rate
        lambd (float): mRNA birth rate
        N (int): Number of Gillespie iterations (default=100_000_000)

    returns: Tuple of mean and var/covars DataFrames for Gillespie sim

    """
    # Adjust model parameters and run Gillespie
    model.tau1 = tau1
    model.alpha = alpha
    model.lambd = lambd
    model.run_gillespie(N)

    # Collect statistics
    means, covs = model.collect_stats()

    # Add columns to DataFrames for model params used in sim
    means["tau1"] = tau1
    means["alpha"] = alpha
    means["lambd"] = lambd
    covs["tau1"] = tau1
    covs["alpha"] = alpha
    covs["lambd"] = lambd

    return means, covs

if __name__ == "__main__":
    random.seed(42)

    # Diffusion matrix
    D = np.array([[1, -1, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1]])

    # Intialise model
    model = M(alpha=10, tau1=2, tau2=2, tau3=4, lambd=10, delta=D)

    # Parameters to test
    taus = [0.1, 1, 3, 5, 50]
    alphas = [0.1, 10, 100]
    lambds = [0.1, 1, 5, 10, 50]

    # Create dask graph of desired simulations
    values = [
        gather_stats(model, t, a, lam) for t in taus for a in alphas for lam in lambds
    ]

    # Execute task graph, running independent simulations in parrallel via Dask
    results = compute(*values, scheduler="processes")

    # Concatonate means and covariances dataframes
    all_means = pd.concat([r[0] for r in results])
    all_covs = pd.concat([r[1] for r in results])

    # Write data to files
    all_means.to_csv('all_means.csv', index=False)
    all_covs.to_csv('all_covs.csv', index=False)
