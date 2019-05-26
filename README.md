## Getting started
Python 3.7
```bash
conda install numpy matplotlib tabulate pandas numba dask  
```

## Usage

#### Python: `mRNADynamicsModel` class
```python
from gillespie import mRNADynamicsModel as M

D = np.array([[1, -1, 0, 0, 0, 0], 
              [0, 0, 1, -1, 0, 0], 
              [0, 0, 0, 0, 1, -1]])

model = M(alpha=10, tau1=2, tau2=2, tau3=4, lambd=10, delta=D)
model.run_gillespie(N=10_000_000)
means, covariances = model.collect_stats()
```
#### CLI:
```bash
$ python gillespie.py --alpha=10 --tau1=2 --lambd=4 --iters=10_000_000

# Initialised model: <mRNADynamicsModel alpha: 10, tau1: 2, tau2: 2, tau3: 4, lambd: 4>
# Running Gillespie algorithm for 10,000,000 iters...
#
# component      predicted_mean    gillespie_mean    percent_error
# -----------  ----------------  ----------------  ---------------
# T factor                   20           20.0409        -0.204439
# mRNA 1                    160          160.478         -0.299021
# mRNA 2                    320          320.456         -0.142377 
#
# value                        fdt    gillespie      error
# ---------------------  ---------  -----------  ---------
# var(T factor)          0.05         0.0499271   0.145701
# var(mRNA 1)            0.03125      0.0315285  -0.891047
# var(mRNA 2)            0.0197917    0.0197173   0.375834
# cov(T factor, mRNA 1)  0.025        0.0253388  -1.35536
# cov(T factor, mRNA 2)  0.0166667    0.0168648  -1.18876
# cov(mRNA 1, mRNA 2)    0.0194444    0.0196701  -1.16031 
# 
# Gillespie took 9707.35 ms to complete.🐥
```
