# README

## Introduction

This repository provides Python implementations of key algorithms and feasibility analysis techniques for the forthcoming paper:

**"On the Properties of the Metalog Distribution"**
by Manel Baucells, Lonnie Chrisman, Thomas W. Keelin, and Stephen Xu

The code offers tools to fit always-feasible Metalog 2.0 distributions, perform feasibility checks, compute moment statistics (mean, variance, skewness, kurtosis), and identify modes, anti-modes, and roots. This repository is intended as a computational companion to the paper, enabling researchers and practitioners to explore Metalog distribution properties and apply them in modeling and optimization tasks.

## Overview

This repository contains two Python scripts:

* **`algo_astar.py`**: Computes the optimal coefficient vector $a^*$ for a Metalog distribution using constrained quadratic programming with Newton’s method and grid search for feasibility checking.
* **`feasibility_stats.py`**: Provides functions for feasibility checking, statistical moment calculation, and auxiliary computations, which support `algo_astar.py` and can be used independently.

Together, these scripts allow you to fit Metalog distributions, verify feasibility, and compute detailed statistics for analysis or reporting.

## Dependencies

All dependencies are listed in `requirements.txt`. You can install them all at once by running:

```sh
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
numpy>=1.26.0,<2.0
pandas>=2.0.0,<3.0
scipy>=1.11.0,<2.0
cvxopt>=1.3.0,<2.0
sympy>=1.12,<2.0
mpmath>=1.3.0,<2.0
```

Alternatively, you may install the packages individually:

```sh
pip install numpy pandas scipy cvxopt sympy mpmath
```

## Files and Functions

### `algo_astar.py`

Implements Algorithm 2 of the paper, fitting a feasible Metalog 2.0 distribution to `(x, y)` quantile-probability pairs.

**Main Function**
`find_a_star(k, x, y, tol=1e-5, epsilon=1e-5)`
Returns the optimal feasible coefficient vector $a^*$ and associated diagnostics (RSS, mean, variance, standard deviation, modes, antimodes, runtime).

**Supporting Functions** include:

* `calculate_Y(y, k)`: Build Metalog basis matrix
* `grid_search_newtons_method(...)`: Locate potential infeasible points
* `C_matrix(...)`: Build constraint matrix for QP
* `function_G`, `function_G_prime`, `G_value`: Define feasibility function and derivative

Example:

```python
import numpy as np
from algo_astar import find_a_star, calculate_Y

x = np.array([1, 2, 4, 8, 12])
y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
k = 4

a_star_result = find_a_star(k, x, y)
a_star = a_star_result["Best a*"]
print(a_star_result)
```

### `feasibility_stats.py`

Contains feasibility checking and statistical computation utilities.

**Main Functions**

* `feasible(a, tol=1e-6)`: Check density feasibility (interior and tail conditions)
* `summary_stats(a)`: Compute mean, variance, standard deviation, skewness, kurtosis, and report modes/anti-modes

**Supporting Functions** implement moment integrals, root finding, and boundary analysis.

Example:

```python
from feasibility_stats import feasible, summary_stats

a = (22.62, 5.64, 3.19, 35.51)
print(feasible(a))
print(summary_stats(a))
```

## Special Note

To compute reliable `summary_stats`, you should first find a **feasible** coefficient vector $a^*$ using `find_a_star` or manually verify feasibility using `feasible(a)`. Calling `summary_stats` on an infeasible $a$ may yield misleading or undefined results.

## Notes on Usage

* **Tolerance (`tol`)** controls precision of feasibility checks and root finding.
* **Epsilon (`epsilon`)** in `find_a_star` enforces stricter inequality constraints for feasibility. If `find_a_star` stops with an error after 1000 iterations, increase `epsilon` relative to `tol`. The error occurs when a fitted tail value `s(0)` or `s(1)` lies within rounding error of `tol`, for example with `tol=epsilon=1e-6` in the `algo_astar.py` example above.
* The distribution is feasible only if its density is non-negative over `[0, 1]`.

## License

This code is released under the MIT License. See [`LICENSE`](LICENSE) for the full text. You may use, copy, modify, and distribute the code, including for commercial purposes, provided that the copyright notice and the permission notice are included in all copies or substantial portions of it. If you use or modify the code, we kindly request that you cite:

**"On the Properties of the Metalog Distribution"**
by Manel Baucells, Lonnie Chrisman, Thomas W. Keelin, and Stephen Xu (forthcoming)
