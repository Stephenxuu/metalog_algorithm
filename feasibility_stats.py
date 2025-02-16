import math
import numpy as np
from scipy.optimize import root_scalar
from scipy.special import factorial
from scipy.optimize import newton
import sympy as sp
import numpy as np
from math import log,comb
import pandas as pd

# Step 0: Define functions
# 1. b_{t,u} calculation

def round_list(lst):
    """Utility function to round each element in a list to six decimal places."""
    return [round(x, 6) for x in lst]

# Compute b_tu based on the formulas
def compute_b_matrix(k,num_s):
    K=(k+1)//2
    T_max=num_s
    U_max=max(2,K)+num_s-1
    I_max=max(K,2)+1
    
    b = np.zeros((T_max, U_max, I_max+3))  # Initialize matrix to store values
    
    # Function to compute the signed Stirling numbers of the first kind
    def stirling_first_kind(w, n):
        if w == 0 and n == 0:
            return 1
        if w == 0 or n == 0:
            return 0
        if n > w:
            return 0
        
        # Create a matrix to store values
        S = np.zeros((w + 1, n + 1))
        S[0,0] = 1
        
        # Fill the matrix using recurrence relation
        for i in range(1, w + 1):
            for j in range(1, min(i + 1, n + 1)):
                S[i,j] = -(i-1) * S[i-1,j] + S[i-1,j-1]
        
        return int(S[w,n])

    # Function to compute P_w(i,j) based on Stirling numbers
    def P_w(w, i, j):
        if i == j:
            # Handle the case where i = j using limits
            result = 0
            for n in range(1, w + 1):
                s_wn = stirling_first_kind(w, n)
                result += s_wn * n * (i ** (n-1))
            return result
    
        result = 0
        for n in range(1, w + 1):
            s_wn = stirling_first_kind(w, n)
            result += s_wn * (i**n - j**n) / (i - j)
        return result
        
   # Loop over all possible values of t, u, and i to compute b[t, u, i]
    for t in range(T_max):
        for i in range(1,I_max):
            for u in range(i+num_s-1):
                b_tu = 0  # Initialize scalar value for this particular t, u, i
                if u <= t:
                    # Case when u <= t
                    for j in range(0, min(u, i-1) + 1):
                        for v in range(0, min(u-j, i-1) + 1):
                            P_term = P_w(i-v, j, i)
                            factor = (math.factorial(i) * math.factorial(t) *
                                      (-1)**(j+1+i-v) * (-0.5)**(t-u+v) /
                                      (math.factorial(j) * math.factorial(i-v) *
                                       math.factorial(t-u+v) * math.factorial(u-j-v)))
                            b_tu += P_term * factor
                else:
                    # Case when u > t
                    for j in range(0, min(t, i-1) + 1):
                        for v in range(0, min(t-j, i+t-u-1) + 1):
                            P_term = P_w(i+t-u-v, j, i)
                            factor = (math.factorial(i) * math.factorial(t) *
                                      (-1)**(j+1+i+t-u-v) * (-0.5)**v /
                                      (math.factorial(j) * math.factorial(i+t-u-v) *
                                       math.factorial(v) * math.factorial(t-j-v)))
                            b_tu += P_term * factor
                
                # Store the computed value in the matrix
                b[t][u][i] = b_tu
    return b

# 2. Function of M(i)(y),S(i)(y) and mu(i)(y)
# The representation for i-th M function or its equivalent term with the same roots
def M(i, check_a, hat_a, b, y):
    num_mu = len(check_a)
    num_s = len(hat_a)
    K=max(num_mu,num_s)
    # Compute M(i)(y) based on the provided equation for M
     # Ensure y is within the valid range (0, 1)
    if y <= 0 or y >= 1:
        raise ValueError(f"Invalid value for y: {y}. y must be in the range (0, 1).")

    # if s(i)=0
    if all(x==0 for x in hat_a):
        if i<num_mu:
            term_1 = sum(
            check_a[t] * sum(factorial(t) * (-0.5)**(t - i - u) / (factorial(t - i - u) * factorial(u)) * y**u 
                            for u in range(t - i + 1)) 
            for t in range(i, num_mu))
        if i>=num_mu:
            term_1=0
        return term_1
    
    # if K<=i
    elif K<=i:
        term_3 = 1 / (y**i * (1 - y)**i) * sum(
            hat_a[t] * sum(b[t, u, i] * y**u for u in range(i)) for t in range(num_s)
        )
        return term_3
    
    # if K>i, we can use polynomial representation in replace of 
    elif K>i:
        term_1 = sum(
        check_a[t] * sum(factorial(t) * (-0.5)**(t - i - u) / (factorial(t - i - u) * factorial(u)) * y**u 
                        for u in range(t - i + 1)) 
        for t in range(i, num_mu))
        term_2 = log(y/(1 - y))*sum(
        hat_a[t] * sum(factorial(t) * (-0.5)**(t - i - u) / (factorial(t - i - u) * factorial(u)) * y**u 
                    for u in range(t - i + 1)) 
        for t in range(i, num_s))
        if num_s==i:
            term_3 = 1 / (y**i * (1 - y)**i) * (sum(y**u *sum(hat_a[t] * b[t, u, i]  for t in range(i)) for u in range(i)))
        if num_s>i:    
            term_3 = 1 / (y**i * (1 - y)**i) * (
        sum(hat_a[t] * sum(b[t, u ,i] * y**u for u in range(t + i)) for t in range(i, num_s))
        + sum(y**u *sum(hat_a[t] * b[t, u, i]  for t in range(i)) for u in range(i)))
        return term_1 + term_2 + term_3

    
def function_M(i, check_a, hat_a, b):
    num_mu = len(check_a)
    num_s = len(hat_a)
    K = max(num_mu, num_s)
    y = sp.Symbol('y')  # Define y as a symbol
    # Case 1: if s(i) = 0
    if all(x == 0 for x in hat_a):
        if i < num_mu:
            f= lambda y: sum(check_a[t] * sum(factorial(t) * (-0.5)**(t - i - u) / (factorial(t - i - u) * factorial(u)) * y**u for u in range(t - i + 1)) for t in range(i, num_mu))
        return f

    # Case 2: if K <= i
    elif K <= i:
        f= lambda y: 1 / (y**i * (1 - y)**i) * sum(sum(hat_a[t] * b[t, u, i] * y**u for u in range(i)) for t in range(num_s))
        return f

    # Case 3: if K > i
    else:
        f_1 = lambda y: sum(
            check_a[t] * sum(factorial(t) * (-0.5)**(t - i - u) / (factorial(t - i - u) * factorial(u)) * y**u 
                            for u in range(t - i + 1)) 
            for t in range(i, num_mu))
        
        f_2 = lambda y: log(y/(1-y))*sum(
            hat_a[t] * sum(factorial(t) * (-0.5)**(t - i - u) / (factorial(t - i - u) * factorial(u)) * y**u 
                            for u in range(t - i + 1)) 
            for t in range(i, num_s))
        
        if num_s == i:
            f_3 = lambda y: 1 / (y**i * (1 - y)**i) * sum(sum(hat_a[t] * b[t, u, i] * y**u 
                                                        for u in range(i)) 
                                                    for t in range(i))
        else:    
            f_3 = lambda y: 1 / (y**i * (1 - y)**i) * (
                sum(hat_a[t] * sum(b[t, u, i] * y**u for u in range(t + i)) 
                    for t in range(i, num_s)) +
                sum(sum(hat_a[t] * b[t, u, i] * y**u 
                        for u in range(i)) 
                    for t in range(i)))
        f = lambda y: f_1(y) + f_2(y) + f_3(y)    
        return f

def S(i, check_a, hat_a, y):
    num_mu = len(check_a)
    num_s = len(hat_a)
    K=max(num_mu,num_s)
    # Compute M(i)(y) based on the provided equation for M
    # if s(i)=0
    if all(x==0 for x in hat_a):
        return 0
    elif K>i:
        term_2 = sum(
        hat_a[t] * sum(factorial(t) * (-0.5)**(t - i - u) / (factorial(t - i - u) * factorial(u)) * y**u 
                       for u in range(t - i + 1)) 
        for t in range(i, num_s))
        return term_2
    elif K<=i:
        return 0

def function_S(i, check_a, hat_a):
    num_mu = len(check_a)
    num_s = len(hat_a)
    K = max(num_mu, num_s)
    y = sp.Symbol('y')  # Define y as a symbol
    # Case 1: if s(i) = 0
    if all(x == 0 for x in hat_a):
        return 0

    # Case 2: if K <= i
    elif K <= i:
        return 0

    # Case 3: if K > i
    else:
        f = lambda y: sum(
            hat_a[t] * sum(factorial(t) * (-0.5)**(t - i - u) / (factorial(t - i - u) * factorial(u)) * y**u 
                            for u in range(t - i + 1)) 
            for t in range(i, num_s))
        return f

def mu(i, check_a, hat_a, y): 
    num_mu = len(check_a)
    num_s = len(hat_a)
    K=max(num_mu,num_s)
    # Compute M(i)(y) based on the provided equation for M
    if num_mu>i:
        term_1 = sum(
        check_a[t] * sum(factorial(t) * (-0.5)**(t - i - u) / (factorial(t - i - u) * factorial(u)) * y**u 
                            for u in range(t - i + 1)) 
        for t in range(i, num_mu))
        return term_1
    elif num_mu<=i:
        return 0

# check limit of M(i)(y) as y approaches 0 and 1 by Lemma 1

def check_at_zero(i,check_a, hat_a, tol):
    """
    Checks the behavior of M^(i)(y) as y -> 0 based on s^(i)(0) = S(0, i).
    Returns 1 if the condition holds (limit goes to +∞), and -1 otherwise (limit goes to -∞).
    """
    for j in range(0, i+1):
        if abs(S(j, check_a, hat_a, 0)) >tol:
            return 1 if (-1)**(i+1-j) * S(j, check_a, hat_a, 0) > 0 else -1
        elif j==i:
            return 1 if (-1)**(i+1-j) * S(j, check_a, hat_a, 0) > 0 else -1
        else:
            continue
def check_at_one(i,check_a, hat_a, tol):
    """
    Checks the behavior of M^(i)(y) as y -> 1 based on s^(i)(1) = S(1, i).
    Returns 1 if the condition holds (limit goes to +∞), and -1 otherwise (limit goes to -∞).
    """
    for j in range(0, i+1):
        if abs(S(j, check_a, hat_a, 1)) >tol:
            return 1 if S(j, check_a, hat_a, 1) > 0 else -1
        elif j==i:
            return 1 if S(j, check_a, hat_a, 1) > 0 else -1
        else:
            continue
        
        
# bisection_newton method        
def bisection_newton(M, M_func, dM_func, a_0, b_0, b, i, check_a, hat_a, tol, max_iter=100):
    """
    Combined bisection and Newton's method to find the root, ensuring results remain within (0, 1).
    """
    LARGE_NUM = 1e100

    # Initial checks for boundary behavior
    if abs(a_0) < tol:  # Close to 0
        M_a = check_at_zero(i, check_a, hat_a, tol) * LARGE_NUM
    else:
        M_a = M(i, check_a, hat_a, b, a_0)
    if abs(b_0 - 1) < tol:  # Close to 1
        M_b = check_at_one(i, check_a, hat_a, tol) * LARGE_NUM
    else:
        M_b = M(i, check_a, hat_a, b, b_0)
    if M_a * M_b >= 0:
        return None  # No root in the interval
    def perform_bisection(a_0, b_0, M_a, M_b, num_bisection=5):
        """
        Perform a fixed number of bisections to refine the interval.
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

    # Initial bisection step
    a_0, b_0, converged = perform_bisection(a_0, b_0, M_a, M_b)
    midpoint = (a_0 + b_0) / 2.0
    if converged:
        return midpoint

    # Alternating Newton and Bisection with error handling
    # Alternating between Newton and bisection methods
    for iteration in range(max_iter):
        try:
            # Apply Newton's method starting from the midpoint
            newton_result = newton(M_func, midpoint, fprime=dM_func, tol=tol, maxiter=max_iter)
            if 0 < newton_result < 1:
                return newton_result  # Valid root found
            if 1-newton_result < tol or newton_result < tol:
                    pass
            else:
                raise ValueError("Newton's method returned a root outside (0, 1).")
        except (RuntimeError, ValueError):
            # If Newton fails, refine interval using bisection
            if abs(a_0) < tol:  # Close to 0
                M_a = check_at_zero(i, check_a, hat_a, tol) * LARGE_NUM
            else:
                M_a = M(i, check_a, hat_a, b, a_0)
            if abs(b_0 - 1) < tol:  # Close to 1
                M_b = check_at_one(i, check_a, hat_a, tol) * LARGE_NUM
            else:
                M_b = M(i, check_a, hat_a, b, b_0)
            if M_a * M_b >= 0:
                return None  # No root in the interval
            a_0, b_0, converged = perform_bisection(a_0, b_0, M_a, M_b)
            midpoint = (a_0 + b_0) / 2.0
            if converged:
                return midpoint
            # If refinement succeeds, retry Newton's method
            try:
                newton_result = newton(M_func, midpoint, fprime=dM_func, tol=tol, maxiter=max_iter)
                if 0 < newton_result < 1:
                    return newton_result
                if 1-newton_result < 100*tol or newton_result < 100*tol:
                    pass
            except (RuntimeError, ValueError):
                continue  # Proceed to next iteration with refined interval

    # Raise error if all attempts fail
    raise RuntimeError(f"Failed to converge within (0, 1) after {max_iter} iterations. Last midpoint: {midpoint}")
# Recursive function to calculate M(i)(y) when K > i
# 0. Let i=K, plug the representation for M(K) and then solve the roots. store them as r_{K,1}, r_{K,2}, ..., r_{K,K}
# 1. i=-1, plug the representation for M(K-1), using lemma 1 to get the solution for lim_{x\to 0}M(k-1)(x) and lim_{x\to 1}M(k-1)(x), stored as r_{K,0} and r_{K,K+1} respectively.
# 2. plug r(K,0)... r(K,K+1) into M(K-1)(y) to get the result and let $I(K-1,j) \in \{-1, 0, +1\}$ be the sign for M(K-1,j).
# 3. In K, for j from 0 to K+1, Check if the sign of I(j)I(j+1)=-1, if yes, we should use bisection to narrow down the interval and then use Newton method to find the root of M(K-1)(y) in the interval (r_{K,j}, r_{K,j+1}), store the root as r_{K-1,i}, 
# repeat this step until we get all the roots, which ares stored as r_{K-1,i}, where i=1,2,.., number of roots in (0,1) for M(K-1)(y).
# 4. Repeat step 1 to 3 until i=2, we get all the roots for M(2)(y) in the interval (0,1), which are stored as r_{2,i}, where i=1,2,...,number of roots in (0,1) for M(2)(y). And these are also known as the inflection points.
# 5. Get the reprenstation for M(1)(y) and plug inflection points r(2,i) into M(1)(y) to get the result, store it as S(i), where i=1,2,...,number of roots in (0,1) for M(2)(y).
# 6. Return the inflection points and the slopes S(i) as the output.

def feasible(a):
    a=round_list(a)
    k=len(a)
    check_a = []  # Coefficients assigned to mu
    hat_a = []    # Coefficients assigned to s
    tol = 1e-6  # Computation tolerance

    # Iterate over the coefficients and assign in "Z" ordering pattern
    for i in range(k):
        if i % 4 == 0 or i % 4 == 3:  # Indices 0, 3, 4, 7, 8, ... -> check_a
            check_a.append(a[i])
        elif i % 4 == 1 or i % 4 == 2:  # Indices 1, 2, 5, 6, 9, 10, ... -> hat_a
            hat_a.append(a[i])


    # Number of mu and s should correspond to the number of check_a and hat_a
    num_mu = len(check_a)
    num_s = len(hat_a)

    # Calculate K based on the number of coefficients
    K = max(num_mu, num_s)
    L =50
    # Initialize matrices r and I
    r = np.zeros((L, L))  # Roots matrix
    I = np.zeros((L, L))  # Signs matrix
    slopes = []  # Slopes for each root
    slopes_inverse = []  # Store 1/slope values
    M_values = []  # Store M values for each root, which is x for cdf

    # compute b matrix
    b=compute_b_matrix(k,num_s)

    # if s=0, we can use the formula to get the roots directly.
    if all(x==0 for x in hat_a):
        coefficients = np.zeros(num_mu-1)
    # Outer sum over t from i to mu_count - 1
        if 2<num_mu:
            for t in range(2, num_mu):
                # Inner sum over u from 0 to t - i
                for u in range(t - 2 + 1):
                    # Compute the coefficient for y^u
                    term = (factorial(t) * (-0.5)**(t - 2 - u)) / (factorial(t - 2 - u) * factorial(u))
                    if u < len(coefficients) and t < len(check_a):
                        coefficients[u] += check_a[t] * term
                    else:
                        print(f"Index out of range: u={u}, t={t}, len(coefficients)={len(coefficients)}, len(hat_a)={len(hat_a)}")
            roots = np.roots(coefficients)
            if len(roots)>0 or roots is not None:
                roots = [root for root in roots if root > tol and root< 1- tol]
            roots.sort()
        elif 2>=num_mu:
            roots = []
        
    # If K<=2, we can use the formula to get the roots directly.
    elif K<=2:
        # Initialize a vector to hold the coefficients for y^0, y^1, ..., y^(j-1)
        i=2
        coefficients0 = np.zeros(i)
        # Loop over u from 0 to j-1
        u=i-1
        while u >= 0:
            t=num_s-1
            while t >= 0:
                coefficients0[u] += hat_a[t] * b[t, u, i]
                t-=1
            u-=1
        coefficients= coefficients0[::-1]
        roots = np.roots(coefficients)
        if len(roots)>0 or roots is not None:
            roots = [root for root in roots if root > tol and root< 1- tol]
        roots.sort()

    # If K>2, we need to recursively solve for M(i)(y) using bisection and Newton's method.
    elif K>2:
    # Step 0: Solve for M(K)(y) and store roots
    # Initialize a vector to hold the coefficients for y^0, y^1, ..., y^(j-1)
        i=K
        coefficients0 = np.zeros(i)
        # Loop over u from 0 to j-1
        u=i-1
        while u >= 0:
            t=num_s-1
            while t >= 0:
                coefficients0[u] += hat_a[t] * b[t, u, i]
                t-=1
            u-=1
        coefficients= coefficients0[::-1]
        r_i = np.roots(coefficients)
        if len(r_i)>0 or r_i is not None:
            r_i = [root for root in r_i if root > tol and root< 1- tol]
        r_i.sort()
        
        # Store initial roots
        r[i, 0] = 0
        r[i, 1:len(r_i)+1] = r_i
        r[i, len(r_i)+1] = 1
        
        i-=1
        while i >= 2:
            # Get boundary signs by Lemma 1
            I[i, 0] = check_at_zero(i, check_a, hat_a, tol)
            I[i ,len(r_i)+1] = check_at_one(i, check_a, hat_a, tol)
            num_prev_roots = len(r_i)
            # Initialize current roots list
            r_i = [] 
            
            # Get signs at previous roots
            if num_prev_roots>0:
                for j in range(1, num_prev_roots + 1):
                    if r[i+1,j]<0 or r[i+1,j]>1:
                        print(r[i+1,],num_prev_roots,i+1==K)
                    M_value = M(i, check_a, hat_a, b, r[i+1,j])
                    if M_value > tol:
                        I[i, j] = 1
                    elif M_value < -tol:
                        I[i, j] = -1
                    else:
                        I[i, j] = 0
                
            # Find roots between sign changes
            j = 0
            while j <= num_prev_roots:
                if I[i, j] == 0:
                    r_i.append(r[i+1, j])
                    j += 1
                elif I[i,j] * I[i,j+1] == -1:
                    # Bisection method and Newton method to find the root
                    M_func = function_M(i, check_a, hat_a, b)
                    dM_func= function_M(i+1, check_a, hat_a, b)
                    root= bisection_newton(M, M_func, dM_func, r[i+1, j], r[i+1, j+1], b, i, check_a, hat_a, tol)
                    # Store the root in the list for r(K-1)
                    if root is not None:
                        r_i.append(root)
                    r_i = [root for root in r_i if root > tol and root< 1- tol]
                    j+=1
                else:
                    j += 1        
            r[i, 0] = 0
            r[i, 1:len(r_i)+1] = r_i
            r[i, len(r_i)+1] = 1
            i -= 1
        # Get final roots (inflection points)
        roots = r[2, 1:len(r_i)+1]

    modes=[]
    anti_modes=[]
    # Calculate slopes
    for root in roots:
        if M(3,check_a, hat_a, b, root)>=0:
            modes.append(root)
        else:
            anti_modes.append(root)
        slope = M(1, check_a, hat_a, b, root)
        slopes.append(slope)
        slopes_inverse.append(1 / slope if abs(slope) > tol else float('inf'))
        M_values.append(M(0, check_a, hat_a, b, root))

    # if there is a negative slope, the model is not interior feasible, otherwise, the model is interior feasible.
    if any(s < -tol for s in slopes):
        interiorfeasible = False
    else:
        interiorfeasible = True

    # Calculate values for s(0), s(1), s'(0), s'(1), µ'(0), µ'(1)
    s_0 = S(0, check_a, hat_a, 0)
    s_1 = S(0, check_a, hat_a, 1)
    s_prime_0 = S(1, check_a, hat_a, 0)
    s_prime_1 = S(1, check_a, hat_a, 1)
    mu_prime_0 = mu(1, check_a, hat_a, 0)
    mu_prime_1 = mu(1, check_a, hat_a, 1)
    
    # Interior feasibility and tail feasibility checks
    tailfeasible_zero = (
        s_0 >tol or 
        (abs(s_0)<= tol and s_prime_0 < -tol) or 
        (abs(s_0)<= tol and abs(s_prime_0)<= tol and mu_prime_0 > tol) or
        (abs(s_0)<= tol and abs(s_prime_0)<= tol and abs(mu_prime_0) <= tol)
    )
    tailfeasible_one = (
        s_1 > tol or 
        (abs(s_1)<= tol and s_prime_1 > tol) or 
        (abs(s_1)<= tol and abs(s_prime_1)<= tol and mu_prime_1 > tol) or
        (abs(s_1)<= tol and abs(s_prime_1)<= tol and abs(mu_prime_1) <= tol)
    )
    
    tailfeasible = tailfeasible_zero and tailfeasible_one
    feasible=interiorfeasible and tailfeasible

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
    
def summary_stats(a):
    ###############################################################################
    # 1. Signed Stirling numbers of the first kind, as given by your definition
    ###############################################################################
    def stirling_first_kind(w, n):
        if w == 0 and n == 0:
            return 1
        if w == 0 or n == 0:
            return 0
        if n > w:
            return 0
        
        # Create a matrix to store values
        S = np.zeros((w + 1, n + 1))
        S[0, 0] = 1
        
        # Fill the matrix using recurrence relation
        for i in range(1, w + 1):
            for j in range(1, min(i + 1, n + 1)):
                S[i, j] = -(i - 1)*S[i - 1, j] + S[i - 1, j - 1]
        
        return int(S[w, n])

    ###############################################################################
    # 2. Define the integrals I(m,0), I(m,1), I(m,2) using the *signed* Stirling
    #    numbers of the first kind in place of s(n+1,2) and s(n+1,3).
    #    For example, in the original formulas, s(n+1,2) and s(n+1,3) appear.
    ###############################################################################

    def I_m_0(m):
        """
        I(m, 0) = (0.5)^m / (m+1),  valid for m even.
        """
        if m % 2 != 0:
            raise ValueError("I(m,0) requires m to be even.")
        return (0.5**m) / (m + 1)

    def I_m_1(m):
        """
        I(m, 1) = sum_{n=1..m} [ C(m,n)*(0.5)^(m-n) * s(n+1,2) / (n+1)! ],  valid for m odd.
        where s(...) is the *signed* Stirling number of the first kind.
        """
        if m % 2 == 0:
            raise ValueError("I(m,1) requires m to be odd.")
        total = 0.0
        for n in range(1, m + 1):
            s_val = stirling_first_kind(n + 1, 2)  # signed Stirling number of the first kind
            total += comb(m, n) * (0.5**(m - n)) * (s_val / factorial(n + 1))
        return total

    def I_m_2(m):
        """
        I(m, 2) = (pi^2 * (0.5)^m) / [3(m+1)]
                + 2 * sum_{n=2..m} [ C(m,n)*(0.5)^(m-n)* s(n+1,3)/(n+1)! ],
        valid for m even.
        """
        if m % 2 != 0:
            raise ValueError("I(m,2) requires m to be even.")
        part1 = (math.pi**2) * (0.5**m) / (3*(m + 1))
        part2 = 0.0
        for n in range(2, m + 1):
            s_val = stirling_first_kind(n + 1, 3)  # signed Stirling number of the first kind
            part2 += comb(m, n) * (0.5**(m - n)) * (s_val / factorial(n + 1))
        return part1 + 2.0 * part2

    ###############################################################################
    #  Mean M = sum_{j≡1 mod 4} a_j I( (j-1)/2, 0 ) + sum_{j≡3 mod 4} a_j I( (j-1)/2, 1 )
    ###############################################################################
    def metalog_mean(a):
        k = len(a)
        mean_val = 0.0
        for j in range(1, k+1):
            aj = a[j-1]
            if j % 4 == 1:
                # j-1 is multiple of 4 => call I_m_0(j-1)
                mean_val += aj * I_m_0((j-1)//2)
            elif j % 4 == 3:
                # j-1 is even but not multiple of 4 => call I_m_1(j-1)
                mean_val += aj * I_m_1((j-1)//2)
        return mean_val

    ###############################################################################
    #  Now the piecewise V_j, V_{j,j'}^{even}, and V_{j,j'}^{odd} definitions.
    #  We will write small helper functions to encode exactly what is in the paper.
    ###############################################################################

    def Vj(j):
        """Implements the piecewise definition of V_j depending on j(mod 4)."""
        r = j % 4
        if r == 1:
            # V_j = I(j-1, 0) - [ I( (j-1)/2, 0 ) ]^2
            # (Note j-1 is multiple of 4 => can call I_m_0(j-1).)
            return I_m_0(j-1) - (I_m_0((j-1)//2))**2
        elif r == 2:
            # V_j = I(j-2, 2).  j-2 is multiple of 4 => can call I_m_2(j-2).
            return I_m_2(j-2)
        elif r == 3:
            # V_j = I(j-1, 2) - [ I((j-1)/2, 1) ]^2
            return I_m_2(j-1) - (I_m_1((j-1)//2))**2
        else:  # r == 0
            # V_j = I(j-2, 0). j-2 is even => can call I_m_0(j-2).
            return I_m_0(j-2)

    def Vjj_even(j, jprime):
        """
        V_{j,j'}^{even}, from the piecewise definition:
        If j,j'(mod4)=2, then I( (j+j'-4)/2, 2 )
        If j,j'(mod4)=0, then I( (j+j'-4)/2, 0 )
        If j(mod4)=0 and j'(mod4)=2 or vice versa, then I( (j+j'-4)/2, 1 )
        """
        # We just check the mod4 patterns:
        rj  = j % 4
        rjp = jprime % 4
        M   = (j + jprime - 4)//2
        if rj == 1 or rjp == 1 or rj == 3 or rjp == 3:
            return 0
        elif rj == 2 and rjp == 2:
            return I_m_2(M)
        elif rj == 0 and rjp == 0:
            return I_m_0(M)
        else:
            # The remaining even pairs are (rj=0, rj'=2) or (rj=2, rj'=0)
            return I_m_1(M)

    def Vjj_odd(j, jprime):
        """
        V_{j,j'}^{odd}, from the piecewise definition:
        1) if j,j'(mod4)=1: I( (j+j'-2)/2, 0 ) - I( (j-1)/2, 0 ) * I( (j'-1)/2, 0 )
        2) if j,j'(mod4)=3: I( (j+j'-2)/2, 2 ) - I( (j-1)/2, 1 ) * I( (j'-1)/2, 1 )
        3) if j(mod4)=1, j'(mod4)=3: I( (j+j'-2)/2, 1 ) - I( (j-1)/2, 0 ) * I( (j'-1)/2, 1 )
        4) if j(mod4)=3, j'(mod4)=1: I( (j+j'-2)/2, 1 ) - I( (j-1)/2, 1 ) * I( (j'-1)/2, 0 )
        """
        rj  = j % 4
        rjp = jprime % 4
        # M = (j+j'-2)//2 for the integral calls below
        M = (j + jprime - 2)//2
        if rj == 0 or rjp == 0 or rj == 2 or rjp == 2:
            return 0
        elif rj == 1 and rjp == 1:
            return I_m_0(M) - I_m_0((j-1)//2)*I_m_0((jprime-1)//2)
        elif rj == 3 and rjp == 3:
            return I_m_2(M) - I_m_1((j-1)//2)*I_m_1((jprime-1)//2)
        elif rj == 1 and rjp == 3:
            return I_m_1(M) - I_m_0((j-1)//2)*I_m_1((jprime-1)//2)
        else: # rj=3, rjp=1
            return I_m_1(M) - I_m_1((j-1)//2)*I_m_0((jprime-1)//2)

    ###############################################################################
    # Putting it all together: the *complete* Metalog variance from the paper’s
    # piecewise definitions.  Then std. dev. is just sqrt.
    ###############################################################################
    def metalog_variance(a):
        """
        Implements the paper's formula for Var[M]:
        Var[M] = sum_{j=2..k} a_j^2 * V_j
                    + sum_{2 <= j < j' <= k, j,j' even} 2 a_j a_{j'} V_{j,j'}^{even}
                    + sum_{3 <= j < j' <= k, j,j' odd} 2 a_j a_{j'} V_{j,j'}^{odd}.
        This *already* accounts for the needed cross terms and for subtracting E[M]^2.
        """
        k = len(a)
        var_val = 0.0

        # 1) Sum_{j=2..k} a_j^2 * V_j
        for j in range(2, k+1):
            var_val += (a[j-1]**2)*Vj(j)

        # 2) Sum_{2 <= j < j' <= k, j,j' even} 2 a_j a_{j'} V_{j,j'}^{even}
        #    "j even" in 1-based means j%2==0 => j=2,4,6,...
        for j in range(2, k):
            for jprime in range(j+1, k+1):
                if (j % 2 == 0) and (jprime % 2 == 0):
                    var_val += 2.0*a[j-1]*a[jprime-1]*Vjj_even(j, jprime)

        # 3) Sum_{3 <= j < j' <= k, j,j' odd} 2 a_j a_{j'} V_{j,j'}^{odd}
        #    "j odd" in 1-based means j%2==1 => j=3,5,7,...
        for j in range(3, k):
            for jprime in range(j+1, k+1):
                if (j % 2 == 1) and (jprime % 2 == 1):
                    var_val += 2.0*a[j-1]*a[jprime-1]*Vjj_odd(j, jprime)

        return var_val

    def metalog_std(a):
        """Standard deviation = sqrt( the complete variance )."""
        return math.sqrt(metalog_variance(a))
    
    mu = metalog_mean(a)
    var = metalog_variance(a)
    sd  = metalog_std(a)
    
    return {
        "mean": mu,
        "variance": var,
        "standard deviation": sd
    }
    
#sample test
#a=(22.71, 1.74, 486.9, 15.4, -2398)
#check=feasible(a)
#print(check)
#result=summary_stats(a)
#print(result)