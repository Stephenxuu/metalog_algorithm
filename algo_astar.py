# This file implements the algorithm for fitting an always feasible metalog 2.0 distribution to (quantile, probability) data.
# It implements Algorithm 1 from the forthcoming paper:
# "On the Properties of the Metalog Distribution" by Manel Baucells, Lonnie Chrisman, and Thomas W. Keelin
#
# This code is written by Stephen Xu. This code is provided freely for any purpose, including academic, commercial, and personal
# use. There are no restrictions on modification, redistribution, or incorporation into other projects. If you find this useful,
# we kindly request that you cite the above paper.
#
# The main function is find_a_star(k, x, y), which returns the optimal a-coefficients for a k-term metalog distribution
# that best fits the (x,y) points -- i.e., the ( quantile, probability ) points.
#
# Example usage:
# x = np.array( [ 1, 2, 4, 8, 12 ] )              # quantile values
# y = np.array( [ 0.1, 0.3, 0.5, 0.7, 0.9 ] )     # cumulative probabilities
# k = 4                                           # number of terms
# a_star = find_a_star( k, x, y )                 # The optimal metalog coefficients
#
# After you have these optimal a_star coefficients, you can find the quantile values at arbitrary cumulative probability
# levels, p, using:
# p = np.array( [0.2, 0.4, 0.6, 0.8] )            # The probability levels of interest
# basis = calculate_Y( p, k )                     # The basis for the probabilities of interest
# quantiles = np.sum( a_star * basis, axis=1 )    # The desired quantile

from feasibility_stats import feasible as fb
from feasibility_stats import compute_b_matrix as compute_b_matrix
from feasibility_stats import M as M
from feasibility_stats import S as S
from feasibility_stats import function_M as function_M
from feasibility_stats import function_S as function_S
from feasibility_stats import summary_stats as summary_stats
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.special import factorial
from scipy.optimize import newton
from math import log
import sympy as sp
import time
from cvxopt import matrix, solvers

def round_list(lst):
    """
    Rounds each element in a list to six decimal places.

    Args:
        lst (list): List of numbers to be rounded.

    Returns:
        list: List with each element rounded to six decimal places.
    """
    return [round(x, 6) for x in lst]

def function_G(check_a, hat_a, b):
    """
    Returns a callable function representing G(y) = y * (1 - y) * M(1)(y).

    Args:
        check_a (list): Coefficients for mu terms.
        hat_a (list): Coefficients for s terms.
        b (numpy.ndarray): Precomputed b matrix.

    Returns:
        callable: Function G(y) that takes y as input.
    """
    num_mu = len(check_a)
    num_s = len(hat_a)
    K = max(num_mu, num_s)
    i = 1
    if all(x == 0 for x in hat_a):
        if i < num_mu:
            f = lambda y: y * (1 - y) * sum(check_a[t] * sum(factorial(t) * (-0.5) ** (t - i - u) / (factorial(t - i - u) * factorial(u)) * y ** u for u in range(t - i + 1)) for t in range(i, num_mu))
        return f
    elif K <= i:
        f = lambda y: sum(sum(hat_a[t] * b[t, u, i] * y ** u for u in range(i)) for t in range(num_s))
        return f
    else:
        f_1 = lambda y: y * (1 - y) * sum(
            check_a[t] * sum(factorial(t) * (-0.5) ** (t - i - u) / (factorial(t - i - u) * factorial(u)) * y ** u
                             for u in range(t - i + 1))
            for t in range(i, num_mu))
        f_2 = lambda y: log(y / (1 - y)) * sum(
            hat_a[t] * sum(factorial(t) * (-0.5) ** (t - i - u) / (factorial(t - i - u) * factorial(u)) * y ** u
                           for u in range(t - i + 1))
            for t in range(i, num_s))
        if num_s == i:
            f_3 = lambda y: sum(sum(hat_a[t] * b[t, u, i] * y ** u
                                    for u in range(i))
                                for t in range(i))
        else:
            f_3 = lambda y: (
                sum(hat_a[t] * sum(b[t, u, i] * y ** u for u in range(t + i))
                    for t in range(i, num_s)) +
                sum(sum(hat_a[t] * b[t, u, i] * y ** u
                        for u in range(i))
                    for t in range(i)))
        f = lambda y: f_1(y) + f_2(y) + f_3(y)
        return f

def function_G_prime(check_a, hat_a, b):
    """
    Returns a callable function representing the derivative G'(y).

    Args:
        check_a (list): Coefficients for mu terms.
        hat_a (list): Coefficients for s terms.
        b (numpy.ndarray): Precomputed b matrix.

    Returns:
        callable: Function G'(y) = (1 - 2y) * M(1)(y) + y * (1 - y) * M(2)(y).
    """
    num_mu = len(check_a)
    num_s = len(hat_a)
    K = max(num_mu, num_s)
    function_M1 = function_M(1, check_a, hat_a, b)
    function_M2 = function_M(2, check_a, hat_a, b)
    G_prime = lambda y: (1 - 2 * y) * function_M1(y) + y * (1 - y) * function_M2(y)
    return G_prime

def G_value(a, b, y):
    """
    Computes the value of G(y) for given coefficients and y.

    Args:
        a (tuple): Coefficients of the metalog distribution.
        b (numpy.ndarray): Precomputed b matrix.
        y (float): Input value in [0, 1].

    Returns:
        float: Value of G(y).

    Raises:
        ValueError: If y is not in [0, 1].
    """
    check_a = []
    hat_a = []
    k = len(a)
    for j in range(k):
        if j % 4 == 0 or j % 4 == 3:
            check_a.append(a[j])
        elif j % 4 == 1 or j % 4 == 2:
            hat_a.append(a[j])
    if y > 1 or y < 0:
        raise ValueError("y is out of the bound [0,1].")
    elif 0 < y < 1:
        G = y * (1 - y) * M(1, check_a, hat_a, b, y)
    else:
        G = sum(hat_a[t] * (y - 0.5) ** t for t in range(len(hat_a)))
    return G

def grid_search_newtons_method(a, b, tol):
    """
    Identifies infeasible points where G(y) < 0 using grid search and Newton's method.

    Args:
        a (tuple): Coefficients of the metalog distribution.
        b (numpy.ndarray): Precomputed b matrix.
        tol (float): Tolerance for convergence.

    Returns:
        list: List of y values where G(y) < 0 within [0, 1].
    """
    check_a = []
    hat_a = []
    k = len(a)
    for i in range(k):
        if i % 4 == 0 or i % 4 == 3:
            check_a.append(a[i])
        elif i % 4 == 1 or i % 4 == 2:
            hat_a.append(a[i])

    def G(y):
        """
        Computes the feasibility function G(y) for a given y.

        Args:
            y (float): Input value in [0, 1].

        Returns:
            float: Value of G(y), representing y * (1 - y) * M(1)(y) for 0 < y < 1, or a polynomial at boundaries.

        Raises:
            ValueError: If y is not in [0, 1].
        """
        if y > 1 or y < 0:
            raise ValueError("y is out of the bound [0,1].")
        elif 0 < y < 1:
            G = y * (1 - y) * M(1, check_a, hat_a, b, y)
        else:
            G = sum(hat_a[t] * (y - 0.5) ** t for t in range(len(hat_a)))
        return G

    def G_prime(y):
        """
        Computes the first derivative G'(y) for a given y.

        Args:
            y (float): Input value in [0, 1].

        Returns:
            float: Value of G'(y), using (1 - 2y) * M(1)(y) + y * (1 - y) * M(2)(y) for 0 < y < 1, or S(1)(y) at boundaries.

        Raises:
            ValueError: If y is not in [0, 1].
        """
        if y > 1 or y < 0:
            raise ValueError("y is out of the bound [0,1].")
        elif 0 < y < 1:
            G_prime = (1 - 2 * y) * M(1, check_a, hat_a, b, y) + y * (1 - y) * M(2, check_a, hat_a, b, y)
        else:
            G_prime = S(1, check_a, hat_a, y)
        return G_prime

    def G_doubleprime(y):
        """
        Computes the second derivative G''(y) for a given y.

        Args:
            y (float): Input value in [0, 1].

        Returns:
            float: Value of G''(y), using -2 * M(1)(y) + 2 * (1 - 2y) * M(2)(y) + y * (1 - y) * M(3)(y) for 0 < y < 1, or S(2)(y) at boundaries.

        Raises:
            ValueError: If y is not in [0, 1].
        """
        if y > 1 or y < 0:
            raise ValueError("y is out of the bound [0,1].")
        elif 0 < y < 1:
            G_doubleprime = -2 * M(1, check_a, hat_a, b, y) + 2 * (1 - 2 * y) * M(2, check_a, hat_a, b, y) + y * (1 - y) * M(3, check_a, hat_a, b, y)
        else:
            G_doubleprime = S(2, check_a, hat_a, y)
        return G_doubleprime

    grid_points = np.concatenate([
        np.array([5 * 10**-i for i in range(3, 16)]),
        np.array([10**-i for i in range(3, 16)]),
        np.arange(0, 1.01, 0.01),
        np.array([1 - 5 * 10**-i for i in range(3, 16)]),
        np.array([1 - 10**-i for i in range(3, 16)])
    ])
    grid_points = np.sort(np.unique(grid_points))
    list_y = []
    max_iterations = 100

    for i in range(1, len(grid_points) - 1):
        y_left, y_center, y_right = grid_points[i - 1], grid_points[i], grid_points[i + 1]
        if G(y_center) <= G(y_left) and G(y_center) <= G(y_right):
            y_0 = y_center
            G_prime_func = lambda y: G_prime(y)
            G_dprime_func = lambda y: G_doubleprime(y)
            y_newton = newton(G_prime_func, y_0, fprime=G_dprime_func, tol=tol, maxiter=100)
            if G(y_newton) < 0:
                if y_newton > 0 and y_newton < 1:
                    list_y.append(y_newton)
                else:
                    print("Convergence failed for", y_newton)
    if G(0) < G(10**-15) and G(0) < 0:
        list_y.append(0)
    if G(1) < G(1 - 10**-15) and G(1) < 0:
        list_y.append(1)

    return list_y

def C_matrix(y_list, num_mu, num_s, b):
    """
    Constructs the constraint matrix C for inequality constraints based on y values.

    Args:
        y_list (list): List of y values where G(y) < 0.
        num_mu (int): Number of mu coefficients.
        num_s (int): Number of s coefficients.
        b (numpy.ndarray): Precomputed b matrix.

    Returns:
        list: List of constraint vectors.

    Raises:
        ValueError: If any y in y_list is not in [0, 1].
    """
    k = num_mu + num_s

    def c_interior(num_mu, num_s, b, y):
        """
        Computes the constraint vector c for interior points (0 < y < 1).

        Args:
            num_mu (int): Number of mu coefficients.
            num_s (int): Number of s coefficients.
            b (numpy.ndarray): Precomputed b matrix.
            y (float): Input value in (0, 1).

        Returns:
            numpy.ndarray: Constraint vector c for the interior case.
        """
        i = 1
        K = max(num_mu, num_s)
        k = num_mu + num_s
        check_c = np.zeros(num_mu)
        hat_c = np.zeros(num_s)
        for t in range(i, num_mu):
            term = sum(factorial(t) * (-0.5) ** (t - i - u) / (factorial(t - i - u) * factorial(u)) * y ** u * y * (1 - y) for u in range(t - i + 1))
            check_c[t] = term
        if num_s > 0:
            for t in range(0, num_s):
                term1 = log(y / (1 - y)) * sum(factorial(t) * (-0.5) ** (t - i - u) / (factorial(t - i - u) * factorial(u)) * y ** u * y * (1 - y) for u in range(t - i + 1))
                term2 = sum(b[t, u, i] * y ** u for u in range(t + i))
                hat_c[t] += term1
                hat_c[t] += term2
        else:
            pass
        c = np.zeros(k)
        check_idx = 0
        hat_idx = 0
        for i in range(k):
            if i % 4 == 0 or i % 4 == 3:
                c[i] = check_c[check_idx]
                check_idx += 1
            elif i % 4 == 1 or i % 4 == 2:
                c[i] = hat_c[hat_idx]
                hat_idx += 1
        c = np.array(c)
        return c

    def c_corner(num_s, y):
        """
        Computes the constraint vector c for corner points (y = 0 or y = 1).

        Args:
            num_s (int): Number of s coefficients.
            y (float): Input value, either 0 or 1.

        Returns:
            numpy.ndarray: Constraint vector c for the corner case.
        """
        hat_c = np.zeros(num_s)
        if num_s > 0:
            for t in range(0, num_s):
                hat_c[t] = (y - 0.5) ** t
        else:
            pass
        check_c = np.zeros(num_mu)
        c = np.zeros(k)
        check_idx = 0
        hat_idx = 0
        for i in range(k):
            if i % 4 == 0 or i % 4 == 3:
                c[i] = check_c[check_idx]
                check_idx += 1
            elif i % 4 == 1 or i % 4 == 2:
                c[i] = hat_c[hat_idx]
                hat_idx += 1
        return c

    C = []
    for y in y_list:
        if y > 1 or y < 0:
            raise ValueError("y is out of the bound [0,1].")
        elif 0 < y < 1:
            c = c_interior(num_mu, num_s, b, y)
        elif y == 0 or y == 1:
            c = c_corner(num_s, y)
        C.append(c)
    return C

def calculate_Y(y, k):
    """
    Constructs the design matrix Y for the metalog fit.

    Args:
        y (numpy.ndarray): Array of cumulative probabilities.
        k (int): Number of terms in the metalog distribution.

    Returns:
        numpy.ndarray: Y matrix of shape (n, k).
    """
    n = len(y)
    Y = np.zeros((n, k))
    for j in range(k):
        power = j // 2
        if j % 4 == 0 or j % 4 == 3:
            Y[:, j] = (y - 0.5) ** power
        else:
            Y[:, j] = (y - 0.5) ** power * np.log(y / (1 - y))
    return Y

def f(a, x, Y):
    """
    Computes the residual sum of squares for the metalog fit.

    Args:
        a (numpy.ndarray): Coefficients of the metalog distribution.
        x (numpy.ndarray): Quantile values.
        Y (numpy.ndarray): Design matrix.

    Returns:
        float: Residual sum of squares.
    """
    residual = x - Y @ a
    return residual.T @ residual

def find_a_star(k, x, y, tol=10e-6, epsilon=10e-6):
    """
    Finds the optimal metalog coefficients a* that minimize the residual sum of squares while ensuring feasibility.

    Args:
        k (int): Number of terms in the metalog distribution.
        x (numpy.ndarray): Quantile values.
        y (numpy.ndarray): Cumulative probabilities.
        tol (float, optional): Tolerance for numerical convergence and root filtering. Defaults to 1e-6.
        epsilon (float, optional): Threshold for inequality constraints in quadratic programming. Defaults to 1e-6.

    Returns:
        dict: Dictionary containing a_ols, a*, objective values, and summary statistics.
    """
    start_time = time.time()
    Y = calculate_Y(y, k)
    Y_T = Y.T
    YTY_inv = np.linalg.inv(Y_T @ Y)
    a_ols = YTY_inv @ Y_T @ x
    a_ols = tuple(a_ols)
    check_a_ols = []
    hat_a_ols = []
    for i in range(k):
        if i % 4 == 0 or i % 4 == 3:
            check_a_ols.append(a_ols[i])
        elif i % 4 == 1 or i % 4 == 2:
            hat_a_ols.append(a_ols[i])
    num_mu = len(check_a_ols)
    num_s = len(hat_a_ols)
    K = max(num_mu, num_s)
    b = compute_b_matrix(k, num_s)
    feasible = False
    y_list = []
    a_list = []
    Q = matrix(2 * Y.T @ Y)
    c = matrix(-2 * Y.T @ x)
    iterations = 0

    while not feasible:
        if iterations == 0:
            a_current = a_ols
        else:
            G = C_matrix(y_list, num_mu, num_s, b)
            G = -1 * np.array(G)
            G = matrix(G, tc='d')
            h = matrix(np.array([-epsilon] * len(y_list)))
            a_current = solvers.qp(Q, c, G, h)['x']
            a_current = np.array(a_current).flatten()
        if K > 2:
            y_list_i = grid_search_newtons_method(a_current, b, tol)
        elif K <= 2:
            check_a = []
            hat_a = []
            for j in range(k):
                if j % 4 == 0 or j % 4 == 3:
                    check_a.append(a_current[j])
                elif j % 4 == 1 or j % 4 == 2:
                    hat_a.append(a_current[j])
            i = 2
            coefficients0 = np.zeros(i)
            u = i - 1
            while u >= 0:
                t = num_s - 1
                while t >= 0:
                    coefficients0[u] += hat_a[t] * b[t, u, i]
                    t -= 1
                u -= 1
            coefficients = coefficients0[::-1]
            roots = np.roots(coefficients)
            y_list_i = []
            roots = [root for root in roots if root > tol and root < 1 - tol]
            for root in roots:
                if G_value(a_current, b, root) < 0:
                    y_list_i.append(root)
            if len(y_list_i) > 0 or y_list_i is not None:
                y_list_i = [root for root in y_list_i if root > tol and root < 1 - tol]
            y_list_i.sort()
        if y_list_i == []:
            feasible_check = fb(a_current)
            if feasible_check["feasible"]:
                feasible = True
            else:
                if not feasible_check["interior_feasible"]:
                    roots = feasible_check["roots"]
                    slopes = feasible_check["slopes"]
                    for j in range(len(roots)):
                        if slopes[j] < 0:
                            y_list.append(roots[j])
                else:
                    if not feasible_check["tail_feasible_zero"]:
                        y_list.append(0)
                    if not feasible_check["tail_feasible_one"]:
                        y_list.append(1)
        else:
            y_list.extend(y_list_i)
        iterations += 1

    iterations = iterations - 1
    a_star = a_current
    f_a_star = f(a_star, x, Y)
    running_time = time.time() - start_time
    summary = summary_stats(a_star)
    mean = summary["mean"]
    variance = summary["variance"]
    sd = summary["standard deviation"]
    modes = summary["modes"]
    antimodes = summary["anti_modes"]
    return {
        "a_ols": a_ols,
        "f(a_ols)": f(a_ols, x, Y),
        "Best a*": a_star,
        "f(a*)": f_a_star,
        "Mean for a*": mean,
        "Variance for a*": variance,
        "Standard deviation for a*": sd,
        "Modes": modes,
        "Antimodes": antimodes,
        "Iterations": iterations,
        "Running time": running_time
    }

def process_datasets(datasets, k_values, output_file):
    """
    Processes multiple datasets for different k values and saves results to an Excel file.

    Args:
        datasets (dict): Dictionary of datasets with 'x' and 'y' arrays.
        k_values (dict): Dictionary mapping dataset names to lists of k values.
        output_file (str): Path to the output Excel file.

    Returns:
        None: Saves results to the specified Excel file.
    """
    results = []
    for dataset_name, data in datasets.items():
        x = data['x']
        y = data['y']
        for k in k_values[dataset_name]:
            result = find_a_star(k, x, y)
            a_ols = result["a_ols"]
            f_a_ols = result["f(a_ols)"]
            a_star = result["Best a*"]
            f_a_star = result["f(a*)"]
            iterations = result.get("Iterations", None)
            running_time = result["Running time"]
            results.append({
                "Dataset": dataset_name,
                "k": k,
                "a_ols": round_list(a_ols),
                "f(a_ols)": round(f_a_ols, 6),
                "a_star": round_list(a_star),
                "f(a*)": round(f_a_star, 6),
                "Iterations": iterations,
                "Running Time (s)": round(running_time, 6)
            })
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

# Example usage
# x = np.array([1, 2, 4, 8, 12])              # Quantile values
# y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])     # Cumulative probabilities
# k = 4                                        # Number of terms
# a_star = find_a_star(k, x, y)                # The optimal metalog coefficients
# print(a_star)
