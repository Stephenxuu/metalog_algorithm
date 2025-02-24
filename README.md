# README

## Introduction
This repository contains Python implementations of key algorithms and feasibility analysis techniques from the forthcoming paper:

**"On the Properties of the Metalog Distribution"**  
by Manel Baucells, Lonnie Chrisman, and Thomas W. Keelin

The code in this repository was developed by Stephen Xu, based on the results and methods described in the paper. It provides tools for finding optimal Metalog 2.0 coefficients that guarantee feasibility, performing feasibility checks, analytically computing moment statistics (mean, variance, standard deviation), obtaining the Metalog 2.0 basis, and identifying modes, anti-modes, and roots.

This repository serves as a computational companion to the paper, enabling researchers and practitioners to explore the mathematical properties of the Metalog distribution and apply optimization techniques to real-world datasets.

## Overview
This repository contains two Python scripts:

1. **`algo_astar.py`**: Implements an algorithm to compute an optimal coefficient vector `a*` using a constrained quadratic programming approach. It incorporates Newton's method and grid search for feasibility checking, ensuring the Metalog distribution fits the data while remaining feasible.
2. **`feasibility_stats.py`**: Provides auxiliary functions for feasibility checking, statistical calculations, and matrix computations, which are utilized by `algo_astar.py`.

These scripts are designed to analyze and optimize Metalog distributions, assess feasibility constraints, and compute key statistical properties.

## Dependencies
The scripts require the following Python libraries:
- `numpy`: For numerical computations and matrix operations.
- `pandas`: For data manipulation and exporting results to Excel.
- `scipy`: For optimization (Newton's method) and special functions (factorials).
- `cvxopt`: For solving quadratic programming problems in `algo_astar.py`.
- `sympy`: For symbolic mathematics and function definitions.

To install the dependencies, run:
```sh
pip install numpy pandas scipy cvxopt sympy
```

## Files and Functions

### `algo_astar.py`
This file implements Algorithm 1 from the forthcoming paper:

**"On the Properties of the Metalog Distribution"**  
by Manel Baucells, Lonnie Chrisman, and Thomas W. Keelin

It fits an always-feasible Metalog 2.0 distribution to (quantile, probability) data. The code is written by Stephen Xu and is provided freely for any purpose, including academic, commercial, and personal use. There are no restrictions on modification, redistribution, or incorporation into other projects. If you find this useful, we kindly request that you cite the above paper.

#### Main Function
- **`find_a_star(k, x, y, tol=1e-6, epsilon=1e-6)`**:  
  Computes the optimal coefficients `a*` for a `k`-term Metalog distribution that best fits the `(x, y)` points (quantile, probability pairs).  
  - **Args**:
    - `k (int)`: Number of terms in the Metalog distribution.
    - `x (numpy.ndarray)`: Quantile values.
    - `y (numpy.ndarray)`: Cumulative probabilities.
    - `tol (float, optional)`: Tolerance for numerical convergence and root filtering. Defaults to `1e-6`.
    - `epsilon (float, optional)`: Threshold for inequality constraints in quadratic programming. Defaults to `1e-6`.
  - **Returns**: A dictionary with `a_ols`, `f(a_ols)`, `Best a*`, `f(a*)`, statistical moments, modes, anti-modes, iterations, and running time.

#### Supporting Functions
- **`calculate_Y(y, k)`**: Constructs the Metalog basis matrix for given probabilities.
- **`f(a, x, Y)`**: Computes the residual sum of squares for the fit.
- **`grid_search_newtons_method(a, b, tol)`**: Identifies infeasible points where the density may be negative.
- **`C_matrix(y_list, num_mu, num_s, b)`**: Builds the constraint matrix for quadratic programming.
- **`function_G(check_a, hat_a, b)`**, **`function_G_prime(check_a, hat_a, b)`**, **`G_value(a, b, y)`**: Define the feasibility function `G(y)` and its derivative.

#### Example Usage
```python
import numpy as np
from algo_astar import find_a_star, calculate_Y

# Define data
x = np.array([1, 2, 4, 8, 12])           # Quantile values
y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # Cumulative probabilities
k = 4                                    # Number of terms

# Compute optimal coefficients
a_star_result = find_a_star(k, x, y)
a_star = a_star_result["Best a*"]
print(a_star_result)

# Compute quantiles for new probability levels
p = np.array([0.2, 0.4, 0.6, 0.8])       # Probability levels of interest
basis = calculate_Y(p, k)                # Metalog basis
quantiles = np.sum(a_star * basis, axis=1)  # Desired quantiles
print("Quantiles:", quantiles)
```

### `feasibility_stats.py`
This file contains functions for feasibility checks and statistical calculations for the Metalog 2.0 distribution, based on:

**"On the Properties of the Metalog Distribution"**  
by Manel Baucells, Lonnie Chrisman, and Thomas W. Keelin

The code is written by Stephen Xu and is provided freely for any purpose, including academic, commercial, and personal use. There are no restrictions on modification, redistribution, or incorporation into other projects. If you find this useful, we kindly request that you cite the above paper.

#### Main Functions
- **`feasible(a, tol=1e-6)`**:  
  Checks the feasibility of a coefficient vector `a`.  
  - **Args**:
    - `a (tuple)`: Coefficients to check.
    - `tol (float, optional)`: Tolerance for numerical comparisons and root filtering. Defaults to `1e-6`.
  - **Returns**: Dictionary with feasibility flags, roots, modes, anti-modes, and slopes.
- **`summary_stats(a)`**: Computes mean, variance, standard deviation, modes, and anti-modes for a coefficient vector `a`.

#### Supporting Functions
- **`compute_b_matrix(k, num_s)`**: Computes the `b` matrix for polynomial coefficient calculations.
- **`M(i, check_a, hat_a, b, y)`**: Computes the i-th M function (core Metalog function).
- **`S(i, check_a, hat_a, y)`**: Computes the i-th S function (related to density).
- **`mu(i, check_a, hat_a, y)`**: Computes the i-th mu function (moment-related).
- **`function_M(i, check_a, hat_a, b)`**, **`function_S(i, check_a, hat_a)`**: Return callable versions of `M` and `S`.
- **`bisection_newton(...)`**: Hybrid bisection-Newton method for solving polynomial roots.


#### Example Usage
```python
from feasibility_stats import feasible, summary_stats

# Check feasibility
a = (0.1, 0.2, -0.1, 0.3)
feasibility_result = feasible(a, tol=1e-5)  # Custom tolerance
print(feasibility_result)

# Compute statistics
stats_result = summary_stats(a)
print(stats_result)
```

## Notes on Usage
- **Tolerance (`tol`)**: Both `find_a_star` and `feasible` allow customization of the tolerance parameter (`tol`), which controls the precision of root finding and feasibility checks. Adjust this value based on your numerical precision needs (e.g., `1e-5` for less strict, `1e-8` for more strict).
- **Epsilon (`epsilon`)**: In `find_a_star`, the `epsilon` parameter defines the threshold for inequality constraints in the quadratic programming step. Smaller values enforce stricter feasibility at the cost of increased computation time.
- **Feasibility**: The Metalog distribution is feasible if its density is non-negative over `[0, 1]`. The `feasible` function checks this by analyzing roots and boundary behavior.

## License
This code is provided freely for any purpose, including academic, commercial, and personal use. There are no restrictions on modification, redistribution, or incorporation into other projects.

If you find this work useful, we kindly request that you cite the paper:

**"On the Properties of the Metalog Distribution"**  
by Manel Baucells, Lonnie Chrisman, and Thomas W. Keelin (forthcoming)
