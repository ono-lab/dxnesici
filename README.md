# DX-NES-ICI
[DX-NES-ICI]() [1] is a Natural Evolution Strategy (NES) for Mixed-Integer Black-Box Optimization (MI-BBO).
DX-NES-ICI reportedly improves the performance of DX-NES-IC [2], one of the most promising continuous BBO methods, on MI-BBO problems.
Simultaneously, DX-NES-ICI outperforms CMA-ES w. Margin [3], one of the most leading MI-BBO methods.


## Getting Started
### Prerequisites
You need [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) that are the packages for scientific computing.

### Installing
Please install via pip.
```bash
$ pip install dxnesici
```


## Usage
### Problem setting
Set the number of dimensions (dim), that of dimensions of continuous variables (dim_co), and that of dimensions of integer variables (dim_int).
Then, set the domains of an integer variable vector $\bar{\textbf{x}}_\text{int}$ and objective function $f(\bar{\textbf{x}})$, where you must assume that the decision variable is $\bar{\textbf{x}} := \left[\bar{\textbf{x}}^\top_\text{co}, \bar{\textbf{x}}^\top_\text{int}\right]^\top$, $\bar{\textbf{x}}^\top_\text{co}$ is a (dim_co)-dimensional continuous variable vector, and $\bar{\textbf{x}}^\top_\text{int}$ is a (dim_int)-dimensional integer variable vector.
Note that the number of elements of the domain in an integer variable must be greater than or equal to 2.
```python
dim = 40
dim_co = dim // 2
dim_int = dim // 2
domain_int = [list(range(-10, 11)) for _ in range(dim_int)]
def n_int_tablet(x):
    x[:dim_co] *= 100
    np.round(x[dim_co:])
    return np.sum(x**2)
```

### The other inputs
Set initial values of mean vector (m), step size (sigma), and population size (lamb).
Note that lamb should be an even number.
A recommended value 1.0 / (dim * lamb) [3] is given as the minimum marginal probability (margin).
```python
m = np.ones([dim, 1]) * 2.
sigma = 1.0
lamb = 8
margin = 1.0 / (dim * lamb)
```

### Running DX-NES-ICI
Pass variables to construct DXNESICI.
You must pass the maximal number of evaluations and a target evaluation value to run the optimizer.
Return values are a success flag, the best solution in the last generation, and the best evaluation value in the last generation.
```python
dxnesici = DXNESICI(dim_co, domain_int, n_int_tablet, m, sigma, lamb, margin)
success, x_best, f_best = dxnesici.optimize(dim * 1e4, 1e-10)
```


## Reference
1. Koki Ikeda and Isao Ono. 2023. Natural Evolution Strategy for Mixed-Integer Black-Box Optimization. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO ’23). 8 pages. https://doi.org/10.1145/3583131.3590518 [[arXiv]()]

2. Masahiro Nomura, Nobuyuki Sakai, Nobusumi Fukushima, and Isao Ono. 2021. Distance-weighted Exponential Natural Evolution Strategy for Implicitly Constrained Black-Box Function Optimization. In IEEE Congress on Evolutionary Computation (CEC ’21). 1099–1106. https://doi.org/10.1109/CEC45853.2021.9504865

3. Ryoki Hamano, Shota Saito, Masahiro Nomura, and Shinichi Shirakawa. 2022. CMA-ES with Margin: Lower-Bounding Marginal Probability for Mixed-Integer Black-Box Optimization. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO ’22). 639–647. https://doi.org/10.1145/3512290.3528827
