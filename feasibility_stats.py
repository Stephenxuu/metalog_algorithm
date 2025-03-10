# This file contains functions for feasibility checks and calculation of statistics for the metalog 2.0 distribution, taken from the paper:
# "On the Properties of the Metalog Distribution" by Manel Baucells, Lonnie Chrisman, and Thomas W. Keelin
#
# This code is written by Stephen Xu. This code is provided freely for any purpose, including academic, commercial, and personal
# use. There are no restrictions on modification, redistribution, or incorporation into other projects. If you find this useful,
# we kindly request that you cite the above paper.

# Example usage:
# a = (22.71, 1.74, 486.9, 15.4, -2398)           # coefficient vector
# Given coefficients, you can check the feasibility of the metalog using:
# check = feasible(a)                             # Check the feasibility

# a = (22.71, 1.74, 486.9, 15.4, -2398)           # coefficient vector
# Given coefficients, you can also find the mean, variance, standard deviation, modes and antimodes of the metalog using:
# result = summary_stats(a)                       # Find the summary statistics

import math
import numpy as np
from scipy.optimize import root_scalar, newton
from scipy.special import factorial
import sympy as sp
import pandas as pd
from math import log, factorial, pi
from scipy.special import comb
from scipy.integrate import quad


# Utility function
def round_list(lst):
    """
    Rounds each element in a list to six decimal places.

    Args:
        lst (list): List of numbers to be rounded.

    Returns:
        list: List with each element rounded to six decimal places.
    """
    return [round(x, 6) for x in lst]
# stirling_first_kind function
def stirling_first_kind(w, n):
        """
        Computes signed Stirling numbers of the first kind using a recurrence relation.

        Args:
            w (int): Upper index of Stirling number.
            n (int): Lower index of Stirling number.

        Returns:
            int: Signed Stirling number S(w, n).
        """
        if w == 0 and n == 0:
            return 1
        if w == 0 or n == 0:
            return 0
        if n > w:
            return 0
        S = np.zeros((w + 1, n + 1))
        S[0, 0] = 1
        for i in range(1, w + 1):
            for j in range(1, min(i + 1, n + 1)):
                S[i, j] = -(i - 1) * S[i - 1, j] + S[i - 1, j - 1]
        return int(S[w, n])

# Compute b_tu matrix
def compute_b_matrix(k, num_s):
    """
    Computes the b_tu matrix based on given formulas for Stirling numbers and P_w terms.

    Args:
        k (int): Parameter determining the size of the matrix.
        num_s (int): Number of s terms.

    Returns:
        numpy.ndarray: 3D array containing computed b[t, u, i] values.
    """
    K = (k + 1) // 2
    T_max = num_s
    U_max = max(2, K) + num_s - 1
    I_max = max(K, 2) + 1
    b = np.zeros((T_max, U_max, I_max + 3))

    def P_w(w, i, j):
        """
        Computes P_w(i, j) using Stirling numbers, handling the i == j case via limits.

        Args:
            w (int): Parameter for Stirling number computation.
            i (int): First argument of P_w.
            j (int): Second argument of P_w.

        Returns:
            float: Computed P_w(i, j) value.
        """
        if i == j:
            result = 0
            for n in range(1, w + 1):
                s_wn = stirling_first_kind(w, n)
                result += s_wn * n * (i ** (n - 1))
            return result
        result = 0
        for n in range(1, w + 1):
            s_wn = stirling_first_kind(w, n)
            result += s_wn * (i ** n - j ** n) / (i - j)
        return result

    for t in range(T_max):
        for i in range(1, I_max):
            for u in range(i + num_s - 1):
                b_tu = 0
                if u <= t:
                    for j in range(0, min(u, i - 1) + 1):
                        for v in range(0, min(u - j, i - 1) + 1):
                            P_term = P_w(i - v, j, i)
                            factor = (math.factorial(i) * math.factorial(t) *
                                      (-1) ** (j + 1 + i - v) * (-0.5) ** (t - u + v) /
                                      (math.factorial(j) * math.factorial(i - v) *
                                       math.factorial(t - u + v) * math.factorial(u - j - v)))
                            b_tu += P_term * factor
                else:
                    for j in range(0, min(t, i - 1) + 1):
                        for v in range(0, min(t - j, i + t - u - 1) + 1):
                            P_term = P_w(i + t - u - v, j, i)
                            factor = (math.factorial(i) * math.factorial(t) *
                                      (-1) ** (j + 1 + i + t - u - v) * (-0.5) ** v /
                                      (math.factorial(j) * math.factorial(i + t - u - v) *
                                       math.factorial(v) * math.factorial(t - j - v)))
                            b_tu += P_term * factor
                b[t][u][i] = b_tu
    return b

# M, S, and mu functions
def M(i, check_a, hat_a, b, y):
    """
    Computes the i-th M function M(i)(y) based on provided coefficients and b matrix.

    Args:
        i (int): Index of the M function.
        check_a (list): Coefficients for mu terms.
        hat_a (list): Coefficients for s terms.
        b (numpy.ndarray): Precomputed b matrix.
        y (float): Input value in (0, 1).

    Returns:
        float: Value of M(i)(y).

    Raises:
        ValueError: If y is not in (0, 1).
    """
    num_mu = len(check_a)
    num_s = len(hat_a)
    K = max(num_mu, num_s)
    if y <= 0 or y >= 1:
        raise ValueError(f"Invalid value for y: {y}. y must be in the range (0, 1).")
    if all(x == 0 for x in hat_a):
        if i < num_mu:
            term_1 = sum(
                check_a[t] * sum(factorial(t) * (-0.5) ** (t - i - u) / (factorial(t - i - u) * factorial(u)) * y ** u
                                 for u in range(t - i + 1))
                for t in range(i, num_mu))
        if i >= num_mu:
            term_1 = 0
        return term_1
    elif K <= i:
        term_3 = 1 / (y ** i * (1 - y) ** i) * sum(
            hat_a[t] * sum(b[t, u, i] * y ** u for u in range(i)) for t in range(num_s)
        )
        return term_3
    elif K > i:
        term_1 = sum(
            check_a[t] * sum(factorial(t) * (-0.5) ** (t - i - u) / (factorial(t - i - u) * factorial(u)) * y ** u
                             for u in range(t - i + 1))
            for t in range(i, num_mu))
        term_2 = log(y / (1 - y)) * sum(
            hat_a[t] * sum(factorial(t) * (-0.5) ** (t - i - u) / (factorial(t - i - u) * factorial(u)) * y ** u
                           for u in range(t - i + 1))
            for t in range(i, num_s))
        if num_s == i:
            term_3 = 1 / (y ** i * (1 - y) ** i) * (sum(y ** u * sum(hat_a[t] * b[t, u, i] for t in range(i)) for u in range(i)))
        if num_s > i:
            term_3 = 1 / (y ** i * (1 - y) ** i) * (
                sum(hat_a[t] * sum(b[t, u, i] * y ** u for u in range(t + i)) for t in range(i, num_s))
                + sum(y ** u * sum(hat_a[t] * b[t, u, i] for t in range(i)) for u in range(i)))
        return term_1 + term_2 + term_3

def function_M(i, check_a, hat_a, b):
    """
    Returns a callable function representing M(i)(y).

    Args:
        i (int): Index of the M function.
        check_a (list): Coefficients for mu terms.
        hat_a (list): Coefficients for s terms.
        b (numpy.ndarray): Precomputed b matrix.

    Returns:
        callable: Function M(i)(y) that takes y as input.
    """
    num_mu = len(check_a)
    num_s = len(hat_a)
    K = max(num_mu, num_s)
    y = sp.Symbol('y')
    if all(x == 0 for x in hat_a):
        if i < num_mu:
            f = lambda y: sum(check_a[t] * sum(factorial(t) * (-0.5) ** (t - i - u) / (factorial(t - i - u) * factorial(u)) * y ** u for u in range(t - i + 1)) for t in range(i, num_mu))
        return f
    elif K <= i:
        f = lambda y: 1 / (y ** i * (1 - y) ** i) * sum(sum(hat_a[t] * b[t, u, i] * y ** u for u in range(i)) for t in range(num_s))
        return f
    else:
        f_1 = lambda y: sum(
            check_a[t] * sum(factorial(t) * (-0.5) ** (t - i - u) / (factorial(t - i - u) * factorial(u)) * y ** u
                             for u in range(t - i + 1))
            for t in range(i, num_mu))
        f_2 = lambda y: log(y / (1 - y)) * sum(
            hat_a[t] * sum(factorial(t) * (-0.5) ** (t - i - u) / (factorial(t - i - u) * factorial(u)) * y ** u
                           for u in range(t - i + 1))
            for t in range(i, num_s))
        if num_s == i:
            f_3 = lambda y: 1 / (y ** i * (1 - y) ** i) * sum(sum(hat_a[t] * b[t, u, i] * y ** u
                                                                  for u in range(i))
                                                              for t in range(i))
        else:
            f_3 = lambda y: 1 / (y ** i * (1 - y) ** i) * (
                sum(hat_a[t] * sum(b[t, u, i] * y ** u for u in range(t + i))
                    for t in range(i, num_s)) +
                sum(sum(hat_a[t] * b[t, u, i] * y ** u
                        for u in range(i))
                    for t in range(i)))
        f = lambda y: f_1(y) + f_2(y) + f_3(y)
        return f

def S(i, check_a, hat_a, y):
    """
    Computes the i-th S function S(i)(y).

    Args:
        i (int): Index of the S function.
        check_a (list): Coefficients for mu terms.
        hat_a (list): Coefficients for s terms.
        y (float): Input value in (0, 1).

    Returns:
        float: Value of S(i)(y).
    """
    num_mu = len(check_a)
    num_s = len(hat_a)
    K = max(num_mu, num_s)
    if all(x == 0 for x in hat_a):
        return 0
    elif K > i:
        term_2 = sum(
            hat_a[t] * sum(factorial(t) * (-0.5) ** (t - i - u) / (factorial(t - i - u) * factorial(u)) * y ** u
                           for u in range(t - i + 1))
            for t in range(i, num_s))
        return term_2
    elif K <= i:
        return 0

def function_S(i, check_a, hat_a):
    """
    Returns a callable function representing S(i)(y).

    Args:
        i (int): Index of the S function.
        check_a (list): Coefficients for mu terms.
        hat_a (list): Coefficients for s terms.

    Returns:
        callable: Function S(i)(y) that takes y as input or 0 if applicable.
    """
    num_mu = len(check_a)
    num_s = len(hat_a)
    K = max(num_mu, num_s)
    y = sp.Symbol('y')
    if all(x == 0 for x in hat_a):
        return 0
    elif K <= i:
        return 0
    else:
        f = lambda y: sum(
            hat_a[t] * sum(factorial(t) * (-0.5) ** (t - i - u) / (factorial(t - i - u) * factorial(u)) * y ** u
                           for u in range(t - i + 1))
            for t in range(i, num_s))
        return f

def mu(i, check_a, hat_a, y):
    """
    Computes the i-th mu function mu(i)(y).

    Args:
        i (int): Index of the mu function.
        check_a (list): Coefficients for mu terms.
        hat_a (list): Coefficients for s terms.
        y (float): Input value in (0, 1).

    Returns:
        float: Value of mu(i)(y).
    """
    num_mu = len(check_a)
    num_s = len(hat_a)
    K = max(num_mu, num_s)
    if num_mu > i:
        term_1 = sum(
            check_a[t] * sum(factorial(t) * (-0.5) ** (t - i - u) / (factorial(t - i - u) * factorial(u)) * y ** u
                             for u in range(t - i + 1))
            for t in range(i, num_mu))
        return term_1
    elif num_mu <= i:
        return 0

# Limit checking functions
def check_at_zero(i, check_a, hat_a, tol):
    """
    Checks the behavior of M(i)(y) as y approaches 0 based on S(j, 0).

    Args:
        i (int): Index of the M function.
        check_a (list): Coefficients for mu terms.
        hat_a (list): Coefficients for s terms.
        tol (float): Tolerance for comparison.

    Returns:
        int: 1 if limit is +∞, -1 if limit is -∞.
    """
    for j in range(0, i + 1):
        if abs(S(j, check_a, hat_a, 0)) > tol:
            return 1 if (-1) ** (i + 1 - j) * S(j, check_a, hat_a, 0) > 0 else -1
        elif j == i:
            return 1 if (-1) ** (i + 1 - j) * S(j, check_a, hat_a, 0) > 0 else -1
        else:
            continue

def check_at_one(i, check_a, hat_a, tol):
    """
    Checks the behavior of M(i)(y) as y approaches 1 based on S(j, 1).

    Args:
        i (int): Index of the M function.
        check_a (list): Coefficients for mu terms.
        hat_a (list): Coefficients for s terms.
        tol (float): Tolerance for comparison.

    Returns:
        int: 1 if limit is +∞, -1 if limit is -∞.
    """
    for j in range(0, i + 1):
        if abs(S(j, check_a, hat_a, 1)) > tol:
            return 1 if S(j, check_a, hat_a, 1) > 0 else -1
        elif j == i:
            return 1 if S(j, check_a, hat_a, 1) > 0 else -1
        else:
            continue

# Root finding
def bisection_newton(M, M_func, dM_func, a_0, b_0, b, i, check_a, hat_a, tol, max_iter=100):
    """
    Finds a root of M(i)(y) in (0, 1) using a hybrid bisection-Newton method.

    Args:
        M (callable): M(i)(y) function.
        M_func (callable): Symbolic M(i)(y) function.
        dM_func (callable): Derivative of M(i)(y).
        a_0 (float): Lower bound of interval.
        b_0 (float): Upper bound of interval.
        b (numpy.ndarray): Precomputed b matrix.
        i (int): Index of the M function.
        check_a (list): Coefficients for mu terms.
        hat_a (list): Coefficients for s terms.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.

    Returns:
        float: Root within (0, 1), or None if no root exists.

    Raises:
        RuntimeError: If convergence fails after max_iter iterations.
    """
    LARGE_NUM = 1e100
    if abs(a_0) < tol:
        M_a = check_at_zero(i, check_a, hat_a, tol) * LARGE_NUM
    else:
        M_a = M(i, check_a, hat_a, b, a_0)
    if abs(b_0 - 1) < tol:
        M_b = check_at_one(i, check_a, hat_a, tol) * LARGE_NUM
    else:
        M_b = M(i, check_a, hat_a, b, b_0)
    if M_a * M_b >= 0:
        return None

    def perform_bisection(a_0, b_0, M_a, M_b, num_bisection=5):
        """
        Performs bisection to refine the interval containing a root.

        Args:
            a_0 (float): Lower bound.
            b_0 (float): Upper bound.
            M_a (float): M(i)(a_0) value.
            M_b (float): M(i)(b_0) value.
            num_bisection (int): Number of bisection steps.

        Returns:
            tuple: (new_a_0, new_b_0, converged) where converged is True if root found.
        """
        for _ in range(num_bisection):
            m = (a_0 + b_0) / 2.0
            M_m = M(i, check_a, hat_a, b, m)
            if abs(M_m) < tol:
                return m, m, True
            if M_a * M_m < 0:
                b_0 = m
                M_b = M_m
            else:
                a_0 = m
                M_a = M_m
        return a_0, b_0, False

    a_0, b_0, converged = perform_bisection(a_0, b_0, M_a, M_b)
    midpoint = (a_0 + b_0) / 2.0
    if converged:
        return midpoint

    for iteration in range(max_iter):
        try:
            newton_result = newton(M_func, midpoint, fprime=dM_func, tol=tol, maxiter=max_iter)
            if 0 < newton_result < 1:
                return newton_result
            if 1 - newton_result < tol or newton_result < tol:
                pass
            else:
                raise ValueError("Newton's method returned a root outside (0, 1).")
        except (RuntimeError, ValueError):
            if abs(a_0) < tol:
                M_a = check_at_zero(i, check_a, hat_a, tol) * LARGE_NUM
            else:
                M_a = M(i, check_a, hat_a, b, a_0)
            if abs(b_0 - 1) < tol:
                M_b = check_at_one(i, check_a, hat_a, tol) * LARGE_NUM
            else:
                M_b = M(i, check_a, hat_a, b, b_0)
            if M_a * M_b >= 0:
                return None
            a_0, b_0, converged = perform_bisection(a_0, b_0, M_a, M_b)
            midpoint = (a_0 + b_0) / 2.0
            if converged:
                return midpoint
            try:
                newton_result = newton(M_func, midpoint, fprime=dM_func, tol=tol, maxiter=max_iter)
                if 0 < newton_result < 1:
                    return newton_result
                if 1 - newton_result < 100 * tol or newton_result < 100 * tol:
                    pass
            except (RuntimeError, ValueError):
                continue
    raise RuntimeError(f"Failed to converge within (0, 1) after {max_iter} iterations. Last midpoint: {midpoint}")

# Feasibility and root computation
def feasible(a, tol=1e-6):
    """
    Determines feasibility of a metalog distribution and computes its roots and properties.

    Args:
        a (tuple): Coefficients of the metalog distribution.
        tol (float, optional): Tolerance for numerical comparisons and root filtering. Defaults to 1e-6.

    Returns:
        dict: Dictionary containing feasibility flags, roots, modes, anti-modes, and slopes.
    """
    a = round_list(a)
    k = len(a)
    check_a = []
    hat_a = []
    for i in range(k):
        if i % 4 == 0 or i % 4 == 3:
            check_a.append(a[i])
        elif i % 4 == 1 or i % 4 == 2:
            hat_a.append(a[i])
    num_mu = len(check_a)
    num_s = len(hat_a)
    K = max(num_mu, num_s)
    L = 50
    r = np.zeros((L, L))
    I = np.zeros((L, L))
    slopes = []
    slopes_inverse = []
    M_values = []
    b = compute_b_matrix(k, num_s)

    if all(x == 0 for x in hat_a):
        coefficients = np.zeros(num_mu - 1)
        if 2 < num_mu:
            for t in range(2, num_mu):
                for u in range(t - 2 + 1):
                    term = (factorial(t) * (-0.5) ** (t - 2 - u)) / (factorial(t - 2 - u) * factorial(u))
                    if u < len(coefficients) and t < len(check_a):
                        coefficients[u] += check_a[t] * term
                    else:
                        print(f"Index out of range: u={u}, t={t}, len(coefficients)={len(coefficients)}, len(hat_a)={len(hat_a)}")
            roots = np.roots(coefficients)
            if len(roots) > 0 or roots is not None:
                roots = [root for root in roots if root > tol and root < 1 - tol]
            roots.sort()
        elif 2 >= num_mu:
            roots = []
    elif K <= 2:
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
        if len(roots) > 0 or roots is not None:
            roots = [root for root in roots if root > tol and root < 1 - tol]
        roots.sort()
    elif K > 2:
        i = K
        coefficients0 = np.zeros(i)
        u = i - 1
        while u >= 0:
            t = num_s - 1
            while t >= 0:
                coefficients0[u] += hat_a[t] * b[t, u, i]
                t -= 1
            u -= 1
        coefficients = coefficients0[::-1]
        r_i = np.roots(coefficients)
        if len(r_i) > 0 or r_i is not None:
            r_i = [root for root in r_i if root > tol and root < 1 - tol]
        r_i.sort()
        r[i, 0] = 0
        r[i, 1:len(r_i) + 1] = r_i
        r[i, len(r_i) + 1] = 1
        i -= 1
        while i >= 2:
            I[i, 0] = check_at_zero(i, check_a, hat_a, tol)
            I[i, len(r_i) + 1] = check_at_one(i, check_a, hat_a, tol)
            num_prev_roots = len(r_i)
            r_i = []
            if num_prev_roots > 0:
                for j in range(1, num_prev_roots + 1):
                    if r[i + 1, j] < 0 or r[i + 1, j] > 1:
                        print(r[i + 1,], num_prev_roots, i + 1 == K)
                    M_value = M(i, check_a, hat_a, b, r[i + 1, j])
                    if M_value > tol:
                        I[i, j] = 1
                    elif M_value < -tol:
                        I[i, j] = -1
                    else:
                        I[i, j] = 0
            j = 0
            while j <= num_prev_roots:
                if I[i, j] == 0:
                    r_i.append(r[i + 1, j])
                    j += 1
                elif I[i, j] * I[i, j + 1] == -1:
                    M_func = function_M(i, check_a, hat_a, b)
                    dM_func = function_M(i + 1, check_a, hat_a, b)
                    root = bisection_newton(M, M_func, dM_func, r[i + 1, j], r[i + 1, j + 1], b, i, check_a, hat_a, tol)
                    if root is not None:
                        r_i.append(root)
                    r_i = [root for root in r_i if root > tol and root < 1 - tol]
                    j += 1
                else:
                    j += 1
            r[i, 0] = 0
            r[i, 1:len(r_i) + 1] = r_i
            r[i, len(r_i) + 1] = 1
            i -= 1
        roots = r[2, 1:len(r_i) + 1]

    modes = []
    anti_modes = []
    for root in roots:
        if M(3, check_a, hat_a, b, root) >= 0:
            modes.append(root)
        else:
            anti_modes.append(root)
        slope = M(1, check_a, hat_a, b, root)
        slopes.append(slope)
        slopes_inverse.append(1 / slope if abs(slope) > tol else float('inf'))
        M_values.append(M(0, check_a, hat_a, b, root))

    interiorfeasible = not any(s < -tol for s in slopes)
    s_0 = S(0, check_a, hat_a, 0)
    s_1 = S(0, check_a, hat_a, 1)
    s_prime_0 = S(1, check_a, hat_a, 0)
    s_prime_1 = S(1, check_a, hat_a, 1)
    mu_prime_0 = mu(1, check_a, hat_a, 0)
    mu_prime_1 = mu(1, check_a, hat_a, 1)
    tailfeasible_zero = (
        s_0 > tol or
        (abs(s_0) <= tol and s_prime_0 < -tol) or
        (abs(s_0) <= tol and abs(s_prime_0) <= tol and mu_prime_0 > tol) or
        (abs(s_0) <= tol and abs(s_prime_0) <= tol and abs(mu_prime_0) <= tol)
    )
    tailfeasible_one = (
        s_1 > tol or
        (abs(s_1) <= tol and s_prime_1 > tol) or
        (abs(s_1) <= tol and abs(s_prime_1) <= tol and mu_prime_1 > tol) or
        (abs(s_1) <= tol and abs(s_prime_1) <= tol and abs(mu_prime_1) <= tol)
    )
    tailfeasible = tailfeasible_zero and tailfeasible_one
    feasible = interiorfeasible and tailfeasible

    return {
        "coefficient": a,
        "roots": roots,
        "modes": modes,
        "anti_modes": anti_modes,
        "slopes": slopes,
        "feasible": feasible,
        "interior_feasible": interiorfeasible,
        "tail_feasible_zero": tailfeasible_zero,
        "tail_feasible_one": tailfeasible_one
    }
 
def I(j, u):
    """
    Numerically approximate I(j,u) = ∫_{0 to 1} (y - 0.5)^j * [ln(y)]^u dy.
    j, u should be nonnegative integers for the formula to make sense
    (and to avoid issues with branch cuts of ln(y) and integrability).
    """

    def integrand(y):
        # Guard against y=0 in the log
        if y == 0.0:
            return 0.0
        return (y - 0.5)**j * (log(y/(1-y)))**u

    # if j and u have different parity: the integral is 0
    if (j % 2) != (u % 2):
        result = 0.0
    # if j and u have the same parity: integrate from 0 to 1
    else:    
        result, err = quad(integrand, 0.0, 1.0)
    return result
    
def generate_combinations(n, length):
    """
    Generate all possible tuples of non-negative integers summing to n with given length.
    Returns a list of tuples, where each tuple represents a valid combination.
    """
    if length == 1:
        if n >= 0:
            return [(n,)]
        return []
    result = []
    for i in range(n + 1):
        for sub_combo in generate_combinations(n - i, length - 1):
            result.append((i,) + sub_combo)
    return result

def raw_moment(t, a):
    """
    Compute the t-th moment of M(y) for given t, k, and coefficients a.
    
    Parameters:
    - t: The moment order (integer ≥ 1)
    - k: Number of terms in M(y) (integer ≥ 2)
    - a: List of coefficients a[j] for j = 1 to k
    
    Returns:
    - The t-th moment value
    """
    
    def floor_div_2(j):
        """
        Compute ⌊(j-1)/2⌋ for a given j.
        """
        return (j - 1) // 2
    
    # Get sets μ and s
    k=len(a)
    mu = [j for j in range(1, k + 1) if j % 4 <= 1]
    s = [j for j in range(1, k + 1) if j % 4 >= 2]
    # Initialize total moment
    total = 0
    
    # Iterate over all possible n1, n2 where n1 + n2 = t
    for n1 in range(t + 1):  # n1 from 0 to t
        n2 = t - n1  # n2 = t - n1
        
        # Generate all combinations of c_j for j in μ summing to n1
        c_combinations = generate_combinations(n1, len(mu))
        
        # Generate all combinations of d_j for j in s summing to n2
        d_combinations = generate_combinations(n2, len(s))
        
        # Iterate over all pairs of c and d combinations
        for c in c_combinations:
            for d in d_combinations:
                # Create dictionaries for c_j and d_j
                c_dict = {mu[i]: c[i] for i in range(len(mu))}
                d_dict = {s[i]: d[i] for i in range(len(s))}
                
                # Compute factorials in denominators
                denom_mu = 1
                for cj in c_dict.values():
                    denom_mu *= factorial(cj)
                denom_s = 1
                for dj in d_dict.values():
                    denom_s *= factorial(dj)
                denom = denom_mu * denom_s
                
                # Compute products in numerators
                prod_mu = 1
                for j in mu:
                    prod_mu *= a[j-1] ** c_dict[j]  # a[j-1] because a is 0-indexed
                prod_s = 1
                for j in s:
                    prod_s *= a[j-1] ** d_dict[j]
                
                # Compute m (exponent for I(m, u))
                m = 0
                for j in mu:
                    m += c_dict[j] * floor_div_2(j)
                for j in s:
                    m += d_dict[j] * floor_div_2(j)
                
                # u is n2
                u = n2
                
                # Compute the term
                term = (factorial(t) / denom) * prod_mu * prod_s * I(m, u)
                total += term
    return total

def central_moment(t, a):
    """
    Computes the t-th central moment E[(M - E[M])^t].
    Args:
        t (int): Order of the central moment (t >= 0).
        a (float): Parameter of the distribution.
    Returns:
        float: The t-th central moment.
    Raises:
        ValueError: If t is negative.
    """
    if not isinstance(t, int) or t < 0:
        raise ValueError("Moment order t must be a non-negative integer.")
    
    # Get the mean (1st raw moment)
    mu = raw_moment(1, a)
    
    if t == 0:
        return 1.0  # E[(M - mu)^0] = 1
    if t == 1:
        return 0.0  # E[M - mu] = 0
    
    # Compute the t-th central moment using raw moments up to t
    central_moment = 0.0
    for k in range(t + 1):
        binom = comb(t, k, exact=True)  # Binomial coefficient
        current_raw_moment = raw_moment(k, a) if k > 0 else 1.0  # E[M^k], E[M^0] = 1
        central_moment += binom * current_raw_moment * (-mu)**(t - k)
    
    return central_moment

def summary_stats(a):
    """
    Computes summary statistics (mean, variance, standard deviation, skewness, kurtosis,
    modes, and anti-modes) for a metalog distribution.

    Args:
        a (tuple): Coefficients of the metalog distribution.

    Returns:
        dict: Dictionary containing mean, variance, standard deviation, skewness, kurtosis,
              modes, and anti-modes.
    """

    # Now compute all desired statistics
    mu = raw_moment(1,a)
    var = central_moment(2,a)
    sd = math.sqrt(var)
    skew = central_moment(3,a) / (sd ** 3)
    kurt = central_moment(4,a) / (sd ** 4)

    # Suppose you have some feasible(...) function:
    modes = feasible(a)["modes"]
    anti_modes = feasible(a)["anti_modes"]

    return {
        "mean": mu,
        "variance": var,
        "standard deviation": sd,
        "skewness": skew,
        "kurtosis": kurt,
        "modes": modes,
        "anti_modes": anti_modes
    }

# Sample test
a = (22.71, 1.74, 486.9, 15.4, -2398)
check = feasible(a)
print(check)

a = (22.71, 1.74, 486.9, 15.4, -2398)
result = summary_stats(a)
print(result)
