from __future__ import division
import numpy as np
#import matplotlib.pyplot as pyplot
import time
from tabulate import tabulate
import pandas as pd
#from numba import njit


delta=np.array([[1, -1, 0, 0, 0, 0],
                [0, 0, 1, -1, 0, 0],
                [0, 0, 0, 0, 1, -1]])

N=1000000
T=np.zeros([1,N])
X=np.zeros([delta.shape[0],N])
tsteps=np.zeros([1,N])
t=0

alpha=10
tau1=2
tau2=2
tau3=4
lambd=10

#theoretical steady states
x1ss=alpha*tau1
x2ss=lambd*tau2*x1ss
x3ss=lambd*tau3*x1ss

##Theoretical (from FDT) variances and covariances
var1=1/x1ss
var2=1/x2ss + tau1/(x1ss*(tau1+tau2))
var3=1/x3ss + tau1/(x1ss*(tau1+tau3))

cov12=tau1/(x1ss*(tau1+tau2))
cov13=tau1/(x1ss*(tau1+tau3))
cov23=(tau1*(2*tau2*tau3+tau1*(tau2+tau3)))/(x1ss*(tau1+tau2)*(tau1+tau3)*(tau2+tau3))


#If we start the simulation from the predicted steady states, we will not need to wait for the initial rise in TF/mRNA levels and our simulation therefore converges much more quickly
x=np.array([x1ss,x2ss,x3ss])
randoms=np.random.rand(N,2)

#

start=time.clock()
for i in range(N):
        rates=[alpha, x[0]/tau1, lambd*x[0], x[1]/tau2, lambd*x[0], x[2]/tau3]
        tau=(-1)/np.sum(rates)*np.log(randoms[i,0])
        t=t+tau
        reac=np.sum(np.cumsum(np.true_divide(rates,np.sum(rates)))<randoms[i,1])
        x=x+delta[:,reac]
        T[:,i]=t
        X[:,i]=x
        tsteps[:,i]=tau
end=time.clock()
print("Gillespie took "+str(end-start)+" seconds")



def plotting():
    pyplot.plot(T[0],X[0],label='Transcription factor')
    pyplot.plot(T[0],X[1],label='mRNA 1')
    pyplot.plot(T[0],X[2],label='mRNA 2')
    pyplot.xlabel('Unitless Time')
    pyplot.ylabel('# of molecules')
    pyplot.legend()

#plotting()

#X=np.delete(X,range(100000),1) #removal of the initial rise
#T=np.delete(T,range(100000),1)
#tsteps=np.delete(tsteps,range(100000),1)

## fluxes
R1p=np.full((1,len(X[0])),alpha)[0]
R1m=X[0]/tau1
R2p=lambd*X[0]
R2m=X[1]/tau2
R3p=R2p
R3m=X[2]/tau3

def histograms():
    pyplot.hist((R1p-R1m), 100, facecolor='red', alpha=0.5)
    pyplot.hist((R2p-R2m), 100, facecolor='green', alpha=0.3)
    pyplot.hist((R3p-R3m), 100, facecolor='blue', alpha=0.3)

#histograms()

#Defining the weighted mean, variance and covariance functions

def tw_mean(series): #time weight the mean
    return np.sum(series*tsteps)/np.sum(tsteps)

def tw_var(series): #time weight the variance
    return np.sum(tsteps*(series-tw_mean(series))**2)/np.sum(tsteps)

def tw_w_var(series): #calculate the weighted time-weighted variance
    return tw_var(series)/(tw_mean(series)**2)

def tw_cov(series1,series2): #calculate the time-weighted covariance
    return np.sum(tsteps*(series1-tw_mean(series1))*(series2-tw_mean(series2)))/np.sum(tsteps)

def tw_w_cov(series1,series2): #calculate the weighted time-weighted covariance
    return tw_cov(series1,series2)/(tw_mean(series1)*tw_mean(series2))

def results_table_means():
    print(tabulate([["Transcription factor", x1ss,tw_mean(X[0]),(x1ss-tw_mean(X[0]))/x1ss * 100],
                    ["mRNA 1", x2ss,tw_mean(X[1]),(x2ss-tw_mean(X[1]))/x2ss * 100],
                    ["mRNA 2", x3ss,tw_mean(X[2]),(x3ss-tw_mean(X[2]))/x3ss * 100]],
                    ["Component", "Predicted mean", "Gillespie mean"," %Error"], tablefmt="grid"))

results_table_means()

def results_table_cov():
    print(tabulate([["var(T factor)", var1,tw_w_var(X[0]),(var1-tw_w_var(X[0]))/var1 * 100],
                    ["var(mRNA 1)", var2,tw_w_var(X[1]),(var2-tw_w_var(X[1]))/var2 * 100],
                    ["var(mRNA 2)", var3,tw_w_var(X[2]),(var3-tw_w_var(X[2]))/var3 * 100],
                    ["cov(T fact, mRNA 1)", cov12,tw_w_cov(X[0],X[1]),(cov12-tw_w_cov(X[0],X[1]))/cov12 * 100],
                    ["cov(T fact, mRNA 2)", cov13,tw_w_cov(X[0],X[2]),(cov13-tw_w_cov(X[0],X[2]))/cov13 * 100],
                    ["cov(mRNA 1, mRNA 2)", cov23,tw_w_cov(X[1],X[2]),(cov23-tw_w_cov(X[1],X[2]))/cov23 * 100]],
                    ["Value", "FDT", "Gillespie"," %Error"], tablefmt="grid"))

results_table_cov()
