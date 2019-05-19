import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from tabulate import tabulate # For text printing to terminal
from numba import njit  # For just-in-time compilation to LLVM
import random
import time
import click  # For command line interface


@njit # Numba decorator; compiles function to efficient machine code (LLVM)
def fast_gillespie(x, alpha, tau1, tau2, tau3, lambd, delta, N):
    """Optimised Gillespie algorithm using jit.

    Args:
        x (ndarray(int)): Initial counts for tf, mRNA1, and mRNA2
        alpha (float): transcription factor birth rate
        tau1 (float): transcription factor lifetime
        tau2 (float): mRNA 1 lifetime
        tau3 (float): mRNA 2 lifetime
        lambd (float): mRNA birth rate
        delta (ndarray): 2D diffusion matrix for system
        N (int): number of iterations for Gillespie

    returns:
        X (ndarray(int)): Trace of component counts for each iteration.
        T (ndarray(float)): Time adjusted trace of time during simulation.
        tsteps (ndarray(float)): Time weight trace; duration of time spent in each state.
    """
    # Initialisation
    t = 0
    T = np.zeros(N)
    tsteps = np.zeros(N)
    X = np.zeros((delta.shape[0], N))

    # Simulation
    for i in range(N):
        # Determine rates
        rates = np.array(
            [alpha, x[0] / tau1, lambd * x[0], x[1] / tau2, lambd * x[0], x[2] / tau3]
        )
        summed = np.sum(rates)

        # Determine WHEN state change occurs
        tau = (-1) / summed * np.log(random.random())
        t = t + tau
        T[i] = t
        tsteps[i] = tau

        # Determine WHICH reaction occurs with relative propabilities
        reac = np.sum(np.cumsum(np.true_divide(rates, summed)) < random.random())
        x = x + delta[:, reac]
        X[:, i] = x

    return X, T, tsteps


class mRNADynamicsModel:
    """Model for the dynamics of mRNA for two genes regulated by a transcription factor.
    
    Args:
        alpha (float): transcription factor birth rate
        tau1 (float): transcription factor lifetime
        tau2 (float): mRNA 1 lifetime
        tau3 (float): mRNA 2 lifetime
        lambd (float): mRNA birth rate
        delta (ndarray): 2D diffusion matrix for system

    Attributes:
        alpha (float): transcription factor birth rate
        tau1 (float): transcription factor lifetime
        tau2 (float): mRNA 1 lifetime
        tau3 (float): mRNA 2 lifetime
        lambd (float): mRNA birth rate
        delta (ndarray): 2D diffusion matrix for system
        labels (list(str)): labels for each component in system
    """

    def __init__(self, alpha, tau1, tau2, tau3, lambd, delta):
        self.alpha = alpha
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3
        self.lambd = lambd
        self.delta = delta
        self.labels = ["T factor", "mRNA 1", "mRNA 2"]

    def compute_theoretical(self):
        """Computes theoretical means, variances, and covariances for system.
        
        returns: tuple of means, variances, and covar

        """
        # Theoretical states
        x1 = self.alpha * self.tau1
        x2 = self.lambd * self.tau2 * x1
        x3 = self.lambd * self.tau3 * x1

        # Theoretical variance and covariances
        v1 = 1 / x1
        v2 = 1 / x2 + self.tau1 / (x1 * (self.tau1 + self.tau2))
        v3 = 1 / x3 + self.tau1 / (x1 * (self.tau1 + self.tau3))

        cov12 = self.tau1 / (x1 * (self.tau1 + self.tau2))
        cov13 = self.tau1 / (x1 * (self.tau1 + self.tau3))
        cov23 = (
            self.tau1
            * (2 * self.tau2 * self.tau3 + self.tau1 * (self.tau2 + self.tau3))
        ) / (
            x1
            * (self.tau1 + self.tau2)
            * (self.tau1 + self.tau3)
            * (self.tau2 + self.tau3)
        )

        return [x1, x2, x3], [v1, v2, v3], [cov12, cov13, cov23]

    def run_gillespie(self, N=100_000_000, start_near_ss=False):
        """Runs Gillespie algorithm using model parameters.
        
        Note: Adds results from Gillespie as class attributes.
        
        """
        # Optional flag for starting simulation from predicted steady states when testing. 
        # False by default.
        x = (
            self.compute_theoretical()[0] if start_near_ss else np.ones(3)
        )  # Start from one for each component.
        self.X, self.T, self.tsteps = fast_gillespie(
            np.ceil(x).astype(int),
            self.alpha,
            self.tau1,
            self.tau2,
            self.tau3,
            self.lambd,
            self.delta,
            N,
        )

    def get_fluxes(self):
        """Calculates difference in fluxes for each component.
        
        returns: List containing the flux balances for each component.

        """
        R1p = (
            (np.full((1, len(self.X[0])), self.alpha)[0]) * self.tsteps
        ).sum() / self.tsteps.sum()
        R1m = ((self.X[0] / self.tau1) * self.tsteps).sum() / self.tsteps.sum()
        R2p = ((self.lambd * self.X[0]) * self.tsteps).sum() / self.tsteps.sum()
        R2m = ((self.X[1] / self.tau2) * self.tsteps).sum() / self.tsteps.sum()
        R3p = R2p
        R3m = ((self.X[2] / self.tau3) * self.tsteps).sum() / self.tsteps.sum()
        return [R1p - R1m, R2p - R2m, R3p - R3m]

    def collect_stats(self):
        """Calculates statistics for most recent Gillespie simulation.
        
        returns: Tuple of means DataFrame & covars/vars DataFrame determined from Gillespie trace.

        """
        # Theoretical values
        Xss, Xvars, Xcovs = self.compute_theoretical()

        tw_means = (self.X * self.tsteps).sum(
            1
        ) / self.tsteps.sum()  # Time weighted means
        p_err = (Xss - tw_means) / Xss * 100
        fluxes = self.get_fluxes()
        mean_stats = pd.DataFrame(
            {
                "component": self.labels,
                "predicted_mean": Xss,
                "gillespie_mean": tw_means,
                "percent_error": p_err,
                "flux": fluxes,
            }
        )

        res = self.X - tw_means[:, np.newaxis]  # Residuals
        tw_vars = (self.tsteps * res ** 2).sum(
            1
        ) / self.tsteps.sum()  # Time weighted variances
        tw_w_vars = tw_vars / tw_means ** 2  # Weighted time-weighted variance
        combs = ((0, 1), (0, 2), (1, 2))  # Covariance combinations
        tw_w_covs = [
            ((self.tsteps * (res[c[0]]) * (res[c[1]])).sum() / np.sum(self.tsteps))
            / (tw_means[c[0]] * tw_means[c[1]])
            for c in combs
        ]  # Weighted time-weighted covariance

        labels = [f"var({lab})" for lab in self.labels] + [
            f"cov({self.labels[c[0]]}, {self.labels[c[1]]})" for c in combs
        ]
        fdts = np.array(Xvars + Xcovs)  # Concatenate lists for dataframe
        all_covs = np.array(list(tw_w_vars) + tw_w_covs)
        vars_covars_stats = pd.DataFrame(
            {
                "value": labels,
                "fdt": fdts,
                "gillespie": all_covs,
                "error": (fdts - all_covs) / fdts * 100,
            }
        )

        return mean_stats, vars_covars_stats

    def plot_X_trace(self):
        """Plots trace of Gillespie simulation for each component."""
        pyplot.plot(self.T, self.X[0], label="Transcription factor")
        pyplot.plot(self.T, self.X[1], label="mRNA 1")
        pyplot.plot(self.T, self.X[2], label="mRNA 2")
        pyplot.xlabel("Unitless Time")
        pyplot.ylabel("# of molecules")
        pyplot.legend(loc="best")

    def plot_flux_hist(self):
        """Plots flux histograms for each component of the Gillespie simulation."""
        d1, d2, d3 = self.get_fluxes()
        pyplot.hist(d1, 100, facecolor="red", alpha=0.5)
        pyplot.hist(d2, 100, facecolor="green", alpha=0.3)
        pyplot.hist(d3, 100, facecolor="blue", alpha=0.3)

    def __repr__(self):
        return f"<mRNADynamicsModel alpha: {self.alpha}, tau1: {self.tau1}, tau2: {self.tau2}, tau3: {self.tau3}, lambd: {self.tau3}>"


@click.command()
@click.option("--alpha", default=10)
@click.option("--tau1", default=2)
@click.option("--tau2", default=2)
@click.option("--tau3", default=4)
@click.option("--lambd", default=4)
@click.option("--iters", default=100_000_000)
def main(alpha, tau1, tau2, tau3, lambd, iters):
    """Runs a single Gillespie simulation.
    
    Note: This func is intended as a CLI. All flags are optional.

    Args:
        alpha (float): transcription factor birth rate
        tau1 (float): transcription factor lifetime
        tau2 (float): mRNA 1 lifetime
        tau3 (float): mRNA 2 lifetime
        lambd (float): mRNA birth rate
        iters (int): number of desired iterations for Gillespie
    
    returns: Prints statistics for single run to terminal.
    """
    random.seed(42)
    D = np.array([[1, -1, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1]])
    model = mRNADynamicsModel(
        alpha=alpha, tau1=tau1, tau2=tau2, tau3=tau3, lambd=lambd, delta=D
    )
    print(f"Initialised model: {model}")
    print(f"Running Gillespie algorithm for {iters:,} iters...\n")
    start = time.time()
    model.run_gillespie(iters)
    end = time.time()
    for table in model.collect_stats():
        print(tabulate(table, headers="keys", showindex=False), "\n")
    print(f"Gillespie took {((end-start)*1000):.2f} ms to complete.🐥")


if __name__ == "__main__":
    main()
