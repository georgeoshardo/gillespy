import random
import numpy as np
import pandas as pd
from tabulate import tabulate

from dask import compute, delayed
import dask.multiprocessing
from dask.diagnostics import ProgressBar
ProgressBar().register()

from gillespie import mRNADynamicsModel as M

random.seed(42)

D = np.array([[1, -1, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1]])

model = M(alpha=10, tau1=2, tau2=2, tau3=4, lambd=10, delta=D)

taus = [1, 2, 5, 10, 50]
alphas = [1, 10, 100]
lambds = [1, 2, 5, 10, 50]


@delayed  # Dask decorator to define tasks
def gather_stats(model, tau1, alpha, lambd, N=1_000_000):
    # Change model parameters and run Gillespie
    model.tau1 = tau1
    model.alpha = alpha
    model.lambd = lambd
    model.run_gillespie(N)

    # Collect statistics
    means, covs = model.collect_stats()
    means["tau1"] = tau1
    means["alpha"] = alpha
    means["lambd"] = lambd
    covs["tau1"] = tau1
    covs["alpha"] = alpha
    covs["lambd"] = lambd
    return means, covs


# Create dask graph of desired simulations
values = [
    gather_stats(model, t, a, lam) for t in taus for a in alphas for lam in lambds
]

# Execute task graph, running independent simulations in parrallel
results = compute(*values, scheduler="processes")

# Concatonate means and covariances dataframes
all_means = pd.concat([r[0] for r in results])
all_covs = pd.concat([r[1] for r in results])

all_means.to_csv('all_means.csv', index=False)
all_means.to_csv('all_covs.csv', index=False)

