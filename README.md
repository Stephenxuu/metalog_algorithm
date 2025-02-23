# README

## Introduction
This repository contains Python implementations of key algorithms and feasibility analysis techniques from the forthcoming paper:

"On the Properties of the Metalog Distribution"

by Manel Baucells, Lonnie Chrisman, and Thomas W. Keelin

The code in this repository was developed by Stephen Xu, based on the results and methods described in the paper. It provides tools for finding optimal metalog 2.0 coefficients that guarantee feasibility, for feasibility checking, to analytically compute moment statistics (mean, variance, standard deviation, skewness and kurtosis), to obtain the metalog 2.0 basis, and to find modes, anti-modes and roots.

This repository serves as a computational companion to the paper, allowing researchers and practitioners to explore the mathematical properties of the metalog distribution and apply optimization techniques to real-world datasets.

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
This file implements the algorithm for fitting an always feasible metalog 2.0 distribution to (quantile, probability) data.
It implements Algorithm 1 from the forthcoming paper:
"On the Properties of the Metalog Distribution" by Manel Baucells, Lonnie Chrisman, and Thomas W. Keelin

This code is written by Stephen Xu. This code is provided freely for any purpose, including academic, commercial, and personal use. There are no restrictions on modification, redistribution, or incorporation into other projects. If you find this useful, we kindly request that you cite the above paper.

The main function is find_a_star(k, x, y), which returns the optimal a-coefficients for a k-term metalog distribution that best fits the (x,y) points -- i.e., the ( quantile, probability ) points.

#### Example usage:
```python
x = np.array( [ 1, 2, 4, 8, 12 ] )              # quantile values
y = np.array( [ 0.1, 0.3, 0.5, 0.7, 0.9 ] )     # cumulative probabilities
k = 4                                           # number of terms
a_star = find_a_star( k, x, y )                 # The optimal metalog coefficients

# After you have these optimal a_star coefficients, you can find the quantile values at arbitrary cumulative probability levels, p, using:
p = np.array( [0.2, 0.4, 0.6, 0.8] )            # The probability levels of interest
basis = calculate_Y( p, k )                     # The basis for the probabilities of interest
quantiles = np.sum( a_star * basis, axis=1 )    # The desired quantiles
```

### `feasibility_stats.py`
This file contains functions for feasibility checks and calculation of statistics for the metalog 2.0 distribution, taken from the paper:

"On the Properties of the Metalog Distribution" by Manel Baucells, Lonnie Chrisman, and Thomas W. Keelin

This code is written by Stephen Xu. This code is provided freely for any purpose, including academic, commercial, and personal use. There are no restrictions on modification, redistribution, or incorporation into other projects. If you find this useful, we kindly request that you cite the above paper.

This script contains functions for feasibility checks and statistical calculations:

- `compute_b_matrix(k, num_s)`: Computes the `b` matrix for polynomial coefficient calculations.
- `M(i, check_a, hat_a, b, y)`, `S(i, check_a, hat_a, y)`, `mu(i, check_a, hat_a, y)`: Define key mathematical functions.
- `feasible(a)`: Checks whether a given coefficient vector `a` is feasible.
- `bisection_newton(...)`: Uses bisection and Newton's method to solve polynomial roots.
- `summary_stats(a)`: Computes summary statistics (mean, variance, standard deviation, modes, antimodes) for a given coefficient vector `a`.

# Example Usage:
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

## License
This code is provided freely for any purpose, including academic, commercial, and personal use. There are no restrictions on modification, redistribution, or incorporation into other projects.

If you find this work useful, we kindly request that you cite the paper:

"On the Properties of the Metalog Distribution"

by Manel Baucells, Lonnie Chrisman, and Thomas W. Keelin (forthcoming)

