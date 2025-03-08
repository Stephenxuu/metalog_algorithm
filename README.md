# README

## Introduction
This repository contains Python implementations of key algorithms and feasibility analysis techniques from the forthcoming paper:

**"On the Properties of the Metalog Distribution"**  
by Manel Baucells, Lonnie Chrisman, and Thomas W. Keelin

The code in this repository was developed by Stephen Xu, based on the results and methods described in the paper. It provides tools for finding optimal Metalog 2.0 coefficients that guarantee feasibility, performing feasibility checks, analytically computing moment statistics (mean, variance, standard deviation, skewness, kurtosis), obtaining the Metalog 2.0 basis, and identifying modes, anti-modes, and roots.

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
- `scipy`: For optimization (Newton's method), special functions (factorials), and numerical integration.
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
- **`find_a_star(k, x, y, tol=10e-6, epsilon=10e-6)`**:  
  Computes the optimal coefficients `a*` for a `k`-term Metalog distribution that best fits the `(x, y)` points (quantile, probability pairs) while ensuring feasibility. Starts with an ordinary least squares solution (`a_ols`) and iteratively refines it using constrained quadratic programming if infeasible, until the density is non-negative across `[0, 1]`.  
  - **Args**:  
    - `k (int)`: Number of terms in the Metalog distribution.  
    - `x (numpy.ndarray)`: Quantile values.  
    - `y (numpy.ndarray)`: Cumulative probabilities.  
    - `tol (float, optional)`: Tolerance for numerical convergence and root filtering. Defaults to `10e-6`.  
    - `epsilon (float, optional)`: Threshold for inequality constraints in quadratic programming. Defaults to `10e-6`.  
  - **Returns**: A dictionary containing:  
    - `"a_ols"`: Initial OLS coefficients.  
    - `"f(a_ols)"`: Residual sum of squares for `a_ols`.  
    - `"Best a*"`: Optimal feasible coefficients.  
    - `"f(a*)"`: Residual sum of squares for `a*`.  
    - `"Mean for a*"`: Mean of the distribution.  
    - `"Variance for a*"`: Variance of the distribution.  
    - `"Standard deviation for a*"`: Standard deviation of the distribution.  
    - `"Modes"`: Modes of the distribution.  
    - `"Antimodes"`: Anti-modes of the distribution.  
    - `"Iterations"`: Number of refinement iterations.  
    - `"Running time"`: Execution time in seconds.  

#### Supporting Functions
- **`calculate_Y(y, k)`**: Constructs the Metalog basis matrix for given probabilities.  
- **`f(a, x, Y)`**: Computes the residual sum of squares for the fit.  
- **`grid_search_newtons_method(a, b, tol)`**: Identifies points where the density may be negative using grid search and Newton’s method, including boundary checks at `y = 0` and `y = 1`.  
- **`C_matrix(y_list, num_mu, num_s, b)`**: Builds the constraint matrix for quadratic programming based on infeasible points.  
- **`function_G(check_a, hat_a, b)`**, **`function_G_prime(check_a, hat_a, b)`**, **`G_value(a, b, y)`**: Define the feasibility function `G(y)` (density-related) and its derivative for feasibility checking.  
- **`process_datasets(datasets, k_values, output_file)`**: Processes multiple datasets for various `k` values and saves results to an Excel file (not used in the main algorithm but provided for batch analysis).  

#### Example Usage
```python
import numpy as np
from algo_astar import find_a_star, calculate_Y

# Define data
x = np.array([1, 2, 4, 8, 12])           # Quantile values
y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # Cumulative probabilities
k = 4                                    # Number of terms

# Compute optimal coefficients
a_star_result = find_a_star(k, x, y)  # Customize tolerance: find_a_star(k, x, y, tol=1e-8, epsilon=1e-7)
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
  Assesses whether a Metalog distribution with coefficients `a` is feasible by ensuring its density remains non-negative over `(0, 1)` and checking tail behavior at boundaries.  
  - **Args**:  
    - `a (tuple)`: Coefficients defining the Metalog distribution (e.g., `(22.71, 1.74, 486.9, 15.4, -2398)`).  
    - `tol (float, optional)`: Numerical tolerance for feasibility checks and root filtering. Defaults to `1e-6`.  
  - **Returns**: A dictionary containing:  
    - `"coefficient"`: Rounded input coefficients.  
    - `"roots"`: Points where the density derivative is zero.  
    - `"modes"`: Local maxima of the distribution.  
    - `"anti_modes"`: Local minima of the distribution.  
    - `"slopes"`: Density slopes at roots.  
    - `"feasible"`: Overall feasibility (True if interior and tail conditions are met).  
    - `"interior_feasible"`: True if density is non-negative inside `(0, 1)`.  
    - `"tail_feasible_zero"`: True if density behaves appropriately as `y → 0`.  
    - `"tail_feasible_one"`: True if density behaves appropriately as `y → 1`.  
  - **Method**:  
    Splits coefficients into `check_a` (mu terms) and `hat_a` (s terms), computes roots of the density derivative using polynomial root-finding or a hybrid bisection-Newton method, and evaluates boundary conditions using `S` and `mu` functions.  
- **`summary_stats(a)`**:  
  Computes statistical properties of the Metalog distribution, including mean, variance, standard deviation, skewness, kurtosis, modes, and anti-modes.  
  - **Args**:  
    - `a (tuple)`: Coefficients of the Metalog distribution.  
  - **Returns**: A dictionary with statistical moments and critical points (modes, anti-modes).

#### Supporting Functions
- **`compute_b_matrix(k, num_s)`**: Generates the `b` matrix of polynomial coefficients using Stirling numbers of the first kind.  
- **`M(i, check_a, hat_a, b, y)`**: Evaluates the i-th Metalog function at `y`, combining mu, logit, and polynomial terms.  
- **`S(i, check_a, hat_a, y)`**: Computes the i-th density-related function.  
- **`mu(i, check_a, hat_a, y)`**: Computes the i-th moment-related function.  
- **`function_M(i, check_a, hat_a, b)`**, **`function_S(i, check_a, hat_a)`**: Provide callable versions of `M` and `S` for root-finding.  
- **`bisection_newton(...)`**: Implements a hybrid bisection-Newton method to locate roots within `(0, 1)` with high precision.  
- **`check_at_zero(i, check_a, hat_a, tol)`**, **`check_at_one(i, check_a, hat_a, tol)`**: Analyze limit behavior at boundaries.  
- **`I(m, u)`**: Numerically approximates the integral `I(m, u) = \int_0^1 (y - 0.5)^m (ln(y/(1-y)))^u dy` using `scipy.integrate.quad`, essential for moment calculations; `m` and `u` are non-negative integers.  
- **`generate_combinations(n, length)`**:Generate all possible tuples of non-negative integers summing to n with given length.
- **`raw_moment(t, a)`**: Calculates the t-th raw moment \( E[M^t] \) using `I(m, u)` to integrate terms over `[0, 1]`.  
- **`central_moment(t, a)`**: Computes the t-th central moment \( E[(M - E[M])^t] \) based on raw moments.  

#### Example Usage
```python
from feasibility_stats import feasible, summary_stats

# Check feasibility
a = (22.71, 1.74, 486.9, 15.4, -2398)
feasibility_result = feasible(a)
print("Feasibility Check:", feasibility_result)

# Compute statistics
stats_result = summary_stats(a)
print("Statistics:", stats_result)
```

## Notes on Usage
- **Tolerance (`tol`)**: Both `find_a_star` and `feasible` allow customization of the tolerance parameter (`tol`), which controls the precision of root finding and feasibility checks. Adjust this value based on your numerical precision needs (e.g., `1e-5` for less strict, `1e-8` for more strict).
- **Epsilon (`epsilon`)**: In `find_a_star`, the `epsilon` parameter defines the threshold for inequality constraints in the quadratic programming step. Smaller values enforce stricter feasibility at the cost of increased computation time.
- **Feasibility**: The Metalog distribution is feasible if its density is non-negative over `[0, 1]`. The `feasible` function checks this by analyzing roots, slopes, and boundary behavior using the `M`, `S`, and `mu` functions.

## License
This code is provided freely for any purpose, including academic, commercial, and personal use. There are no restrictions on modification, redistribution, or incorporation into other projects.

If you find this work useful, we kindly request that you cite the paper:

**"On the Properties of the Metalog Distribution"**  
by Manel Baucells, Lonnie Chrisman, and Thomas W. Keelin (forthcoming)
