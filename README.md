# README

## Overview
This repository contains two Python scripts:

1. `algo_astar.py`: Implements an algorithm to compute an optimal coefficient vector `a*` using a constrained quadratic programming approach. It incorporates Newton's method and grid search for feasibility checking.
2. `feasibility_stats.py`: Provides auxiliary functions for feasibility checking, statistical calculations, and matrix computations, used by `algo_astar.py`.

These scripts are designed to analyze and optimize metalog distributions and assess feasibility constraints in optimization problems.

## Dependencies
The scripts require the following Python libraries:
- `numpy`
- `pandas`
- `scipy`
- `cvxopt`
- `sympy`

To install the dependencies, run:
```sh
pip install numpy pandas scipy cvxopt sympy
```

## Files and Functions

### `algo_astar.py`
This script implements the core optimization algorithm to compute the best coefficient vector `a*` given data `(x, y)`. It includes:

- `find_a_star(k, x, y)`: Finds the optimal coefficient vector `a*` for given `k`.
- `grid_search_newtons_method(a, b, tol)`: Performs a grid search and Newton's method to identify infeasible points.
- `calculate_Y(y, k)`: Constructs the matrix `Y` required for least squares approximation.
- `process_datasets(datasets, k_values, output_file)`: Processes multiple datasets and saves results to an Excel file.

### `feasibility_stats.py`
This script contains functions for feasibility checks and statistical calculations:

- `compute_b_matrix(k, num_s)`: Computes the `b` matrix for polynomial coefficient calculations.
- `M(i, check_a, hat_a, b, y)`, `S(i, check_a, hat_a, y)`, `mu(i, check_a, hat_a, y)`: Define key mathematical functions.
- `feasible(a)`: Checks whether a given coefficient vector `a` is feasible.
- `summary_stats(a)`: Computes summary statistics (mean, variance, standard deviation) of the metalog distribution.
- `bisection_newton(...)`: Uses bisection and Newton's method to solve polynomial roots.
- `summary_stats(a)`: Computes summary statistics (mean, variance, standard deviation) for a given coefficient vector `a`.

## Usage

### Finding Optimal Coefficients
To find the best coefficient vector `a*`, use:
```python
from algo_astar import find_a_star
x = np.array([...])  # Input x values
y = np.array([...])  # Input y values
k = 4  # Number of coefficients
result = find_a_star(k, x, y)
print(result)
```

### Checking Feasibility
To check the feasibility of a given coefficient vector `a`, use:
```python
from feasibility_stats import feasible
a = [0.1, 0.2, -0.1, 0.3]
result = feasible(a)
print(result)
```

### Computing Summary Statistics
To compute the mean, variance, and standard deviation for a given coefficient vector `a`, use:
```python
from feasibility_stats import summary_stats
a = [0.1, 0.2, -0.1, 0.3]
result = summary_stats(a)
print(result)
```

### Processing Multiple Datasets
If you have multiple datasets, you can process them using:
```python
from algo_astar import process_datasets

datasets = {
    'dataset1': {'x': np.array([...]), 'y': np.array([...])},
    'dataset2': {'x': np.array([...]), 'y': np.array([...])}
}
k_values = {'dataset1': [4, 6], 'dataset2': [5]}
output_file = 'results.xlsx'
process_datasets(datasets, k_values, output_file)
```

## Output
- The `find_a_star` function returns an optimal coefficient vector `a*`, along with feasibility details and statistical measures.
- Feasibility results include the feasibility status, inflection points, and function slopes.
- Results from `process_datasets` are saved in an Excel file for further analysis.
