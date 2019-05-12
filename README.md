## Getting started
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
python gillespie.py --alpha=10 --tau1=2 --lambd=4 --iters=1000000

# Initialised model: <mRNADynamicsModel alpha: 10, tau1: 2, tau2: 2, tau3: 4, lambd: 4>
# Running Gillespie algorithm for 1000000 iterations...
# 
# component      predicted_mean    gillespie_mean    percent_error
# -----------  ----------------  ----------------  ---------------
# T factor                   20           20.3037       -1.51861
# mRNA 1                    160          157.957         1.27676
# mRNA 2                    320          319.857         0.0445527 
# 
# value                        fdt     gillespie     error
# ---------------------  ---------  ------------  --------
# var(T factor)          0.05        0.0012264     97.5472
# var(mRNA 1)            0.03125     3.04524e-05   99.9026
# var(mRNA 2)            0.0197917   1.69173e-05   99.9145
# cov(T factor, mRNA 1)  0.025      -9.06481e-05  100.363
# cov(T factor, mRNA 2)  0.0166667  -8.84549e-05  100.531
# cov(mRNA 1, mRNA 2)    0.0194444   1.7516e-05    99.9099 
```
