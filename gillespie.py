import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from tabulate import tabulate
from numba import njit
import random
import time
import click


class mRNADynamicsModel:
    def __init__(self, alpha, tau1, tau2, tau3, lambd, delta):
        self.alpha = alpha
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3
        self.lambd = lambd
        self.delta = delta
        self.labels = ["T factor", "mRNA 1", "mRNA 2"]

    def run_gillespie(self, N=100_000_000, start_at_ss=True):
        # If we start the simulation from the predicted steady states,
        # we will not need to wait for the initial rise in TF/mRNA
        # levels and our simulation therefore converges much more quickly
        x = self.compute_theoretical()[0] if start_at_ss else np.ones(3)
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

    def compute_theoretical(self):
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

    def collect_stats(self):
        # Theoretical
        Xss, Xvars, Xcovs = self.compute_theoretical()

        # Mean and Variance Calculations
        # Time weight the mean
        tw_means = (self.X * self.tsteps).sum(1) / self.tsteps.sum()
        # mean Errors
        p_err = (Xss - tw_means) / Xss * 100

        # Variance and Covariance calculations
        # Time weight the variance
        diffs = self.X - tw_means[:, np.newaxis]
        tw_vars = (self.tsteps * diffs ** 2).sum(1) / self.tsteps.sum()
        # Calculate the weighted time-weighted variance
        tw_w_vars = tw_vars / tw_means ** 2
        combs = ((0, 1), (0, 2), (1, 2))  # Covariance combinations
        # Calculate the weighted time-weighted covariance
        tw_w_covs = [
            ((self.tsteps * (diffs[c[0]]) * (diffs[c[1]])).sum() / np.sum(self.tsteps))
            / (tw_means[c[0]] * tw_means[c[1]])
            for c in combs
        ]

        labs = [f"var({lab})" for lab in self.labels]
        labs += [f"cov({self.labels[c[0]]}, {self.labels[c[1]]})" for c in combs]
        fdts = np.array(Xvars + Xcovs)
        all_covs = np.array(list(tw_w_vars) + tw_w_covs)
        fluxes = self.get_fluxes()
        df1 = pd.DataFrame(
            {
                "component": self.labels,
                "predicted_mean": Xss,
                "gillespie_mean": tw_means,
                "percent_error": p_err,
                "flux": fluxes
            }
        )

        df2 = pd.DataFrame(
            {
                "value": labs,
                "fdt": fdts,
                "gillespie": all_covs,
                "error": (fdts - all_covs) / fdts * 100,
            }
        )

        return df1, df2

    def plot_X_trace(self):
        pyplot.plot(self.T, self.X[0], label="Transcription factor")
        pyplot.plot(self.T, self.X[1], label="mRNA 1")
        pyplot.plot(self.T, self.X[2], label="mRNA 2")
        pyplot.xlabel("Unitless Time")
        pyplot.ylabel("# of molecules")
        pyplot.legend(loc="best")

    #def plot_flux_hist(self):

        pyplot.hist((R1p - R1m), 100, facecolor="red", alpha=0.5)
        pyplot.hist((R2p - R2m), 100, facecolor="green", alpha=0.3)
        pyplot.hist((R3p - R3m), 100, facecolor="blue", alpha=0.3)

    def get_fluxes(self):
        R1p = ((np.full((1, len(self.X[0])), self.alpha)[0])*self.tsteps).sum()/self.tsteps.sum()
        R1m = ((self.X[0] / self.tau1)*self.tsteps).sum()/self.tsteps.sum()
        R2p = ((self.lambd * self.X[0])*self.tsteps).sum()/self.tsteps.sum()
        R2m = ((self.X[1] / self.tau2)*self.tsteps).sum()/self.tsteps.sum()
        R3p = R2p
        R3m = ((self.X[2] / self.tau3)*self.tsteps).sum()/self.tsteps.sum()
        return [R1p-R1m, R2p-R2m, R3p-R3m]

    def __repr__(self):
        return f"<mRNADynamicsModel alpha: {self.alpha}, tau1: {self.tau1}, tau2: {self.tau2}, tau3: {self.tau3}, lambd: {self.tau3}>"


@njit
def fast_gillespie(x, alpha, tau1, tau2, tau3, lambd, delta, N):
    t = 0
    T = np.zeros(N)
    tsteps = np.zeros(N)
    X = np.zeros((delta.shape[0], N))
    for i in range(N):
        rates = np.array(
            [alpha, x[0] / tau1, lambd * x[0], x[1] / tau2, lambd * x[0], x[2] / tau3]
        )
        summed = np.sum(rates)
        tau = (-1) / summed * np.log(random.random())
        t = t + tau
        T[i] = t
        tsteps[i] = tau

        reac = np.sum(np.cumsum(np.true_divide(rates, summed)) < random.random())
        x = x + delta[:, reac]
        X[:, i] = x

    return X, T, tsteps


@click.command()
@click.option("--alpha", default=10)
@click.option("--tau1", default=2)
@click.option("--tau2", default=2)
@click.option("--tau3", default=4)
@click.option("--lambd", default=4)
@click.option("--iters", default=100_000_000)
def main(alpha, tau1, tau2, tau3, lambd, iters):
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
