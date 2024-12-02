from feasible import feasible as fb
from feasible import compute_b_matrix as compute_b_matrix
from feasible import M as M
from feasible import S as S
from feasible import function_M as function_M
from feasible import function_S as function_S
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
    """Utility function to round each element in a list to six decimal places."""
    return [round(x, 6) for x in lst]

def function_G(check_a, hat_a, b):
    num_mu = len(check_a)
    num_s = len(hat_a)
    K = max(num_mu, num_s)
    i=1
    # Case 1: if s(i) = 0
    if all(x == 0 for x in hat_a):
        if i < num_mu:
            f= lambda y: y * (1 - y)*sum(check_a[t] * sum(factorial(t) * (-0.5)**(t - i - u) / (factorial(t - i - u) * factorial(u)) * y**u for u in range(t - i + 1)) for t in range(i, num_mu))
        return f

    # Case 2: if K <= i
    elif K <= i:
        f= lambda y: sum(sum(hat_a[t] * b[t, u, i] * y**u for u in range(i)) for t in range(num_s))
        return f

    # Case 3: if K > i
    else:
        f_1 = lambda y: y * (1 - y) * sum(
            check_a[t] * sum(factorial(t) * (-0.5)**(t - i - u) / (factorial(t - i - u) * factorial(u)) * y**u 
                            for u in range(t - i + 1)) 
            for t in range(i, num_mu))
        
        f_2 = lambda y: log(y/(1-y))*sum(
            hat_a[t] * sum(factorial(t) * (-0.5)**(t - i - u) / (factorial(t - i - u) * factorial(u)) * y**u 
                            for u in range(t - i + 1)) 
            for t in range(i, num_s))
        
        if num_s == i:
            f_3 = lambda y: sum(sum(hat_a[t] * b[t, u, i] * y**u 
                                                        for u in range(i)) 
                                                    for t in range(i))
        else:    
            f_3 = lambda y: (
                sum(hat_a[t] * sum(b[t, u, i] * y**u for u in range(t + i)) 
                    for t in range(i, num_s)) +
                sum(sum(hat_a[t] * b[t, u, i] * y**u 
                        for u in range(i)) 
                    for t in range(i)))
        f = lambda y: f_1(y) + f_2(y) + f_3(y)    
        return f
    
def function_G_prime(check_a, hat_a, b):
    num_mu = len(check_a)
    num_s = len(hat_a)
    K = max(num_mu, num_s)
    function_M1 = function_M(1, check_a, hat_a, b)
    function_M2 = function_M(2, check_a, hat_a, b)
    G_prime= lambda y: (1 - 2 * y) * function_M1(y) + y * (1 - y) * function_M2(y)
    return G_prime
# Grid search for the infeasible points using Newton's method
def G_value(a, b, y):
    # Initialize a vector to hold the coefficients for y^0, y^1, ..., y^(j-1)
    check_a = []  # Coefficients assigned to mu
    hat_a = []    # Coefficients assigned to s            
    k=len(a)
    for j in range(k):
        if j % 4 == 0 or j % 4 == 3:  # Indices 0, 3, 4, 7, 8, ... -> check_a
            check_a.append(a[j])
        elif j % 4 == 1 or j % 4 == 2:  # Indices 1, 2, 5, 6, 9, 10, ... -> hat_a
            hat_a.append(a[j])
    if y>1 or y<0:
        raise ValueError("y is out of the bound [0,1].")
    elif 0<y<1:
        G=y* (1-y) * M(1, check_a, hat_a, b, y)
    else:
        G=sum(hat_a[t]* (y-0.5)**t for t in range(len(hat_a)))
    return G

def grid_search_newtons_method(a, b, tol):
    check_a = []
    hat_a = []
    k=len(a)
    for i in range(k):
        if i % 4 == 0 or i % 4 == 3:  # Indices 0, 3, 4, 7, 8, ... -> check_a
            check_a.append(a[i])
        elif i % 4 == 1 or i % 4 == 2:  # Indices 1, 2, 5, 6, 9, 10, ... -> hat_a
            hat_a.append(a[i])
    # Define function G(y) and its derivative G'(y)
    def G(y):
        if y>1 or y<0:
            raise ValueError("y is out of the bound [0,1].")
        elif 0<y<1:
            G=y* (1-y) * M(1, check_a, hat_a, b, y)
        else:
            G=sum(hat_a[t]* (y-0.5)**t for t in range(len(hat_a)))
        return G

    def G_prime(y):
        if y>1 or y<0:
            raise ValueError("y is out of the bound [0,1].")
        elif 0<y<1:
            G_prime= (1-2*y) * M(1, check_a, hat_a, b, y) + y*(1-y)*M(2, check_a, hat_a, b, y)
        else:
            G_prime=S(1, check_a, hat_a, y)
        return G_prime
    def G_doubleprime(y):
        if y>1 or y<0:
            raise ValueError("y is out of the bound [0,1].")
        elif 0<y<1:
            G_doubleprime= -2* M(1, check_a, hat_a, b, y) + 2*(1-2*y) * M(2, check_a, hat_a, b, y) + y*(1-y)*M(3, check_a, hat_a, b, y)
        else:
            G_doubleprime=S(2, check_a, hat_a, y)
        return G_doubleprime
    
    grid_points = np.concatenate([
        np.array([5 * 10**-i for i in range(3, 16)]),  # {5*10^(-i), i = 3, ..., 15}
        np.array([10**-i for i in range(3, 16)]),       # {10^(-i), i = 3, ..., 15}
        np.arange(0, 1.01, 0.01),           # {0, 0.01, ..., 1}
        np.array([1 - 5 * 10**-i for i in range(3, 16)]),  # {1 - 5*10^(-i), i = 3, ..., 15}
        np.array([1 - 10**-i for i in range(3, 16)])       # {1 - 10^(-i), i = 3, ..., 15}
    ])
    
    # Unique sorted grid points for robustness
    grid_points = np.sort(np.unique(grid_points))
    
    # Placeholder for infeasible points
    list_y = []
    # Newton's method parameters
    max_iterations = 100

    # Step 2: Find rough local minimum
    for i in range(1, len(grid_points) - 1):
        y_left, y_center, y_right = grid_points[i - 1], grid_points[i], grid_points[i + 1]
        if G(y_center) <= G(y_left) and G(y_center) <= G(y_right):
            # Step 3: Use rough local minimum as initial guess for Newton's method
            y_0= y_center
            G_prime_func= lambda y: G_prime(y)
            G_dprime_func= lambda y: G_doubleprime(y)
            y_newton = newton(G_prime_func, y_0, fprime=G_dprime_func, tol=tol, maxiter=100)
             # Step 4: Check if it's a real local minimum
            if G(y_newton) < 0:
                # Step 5: Check feasibility and store if infeasible
                if y_newton > 0 and y_newton < 1:
                    list_y.append(y_newton)
                else:
                    print("Convergence failed for", y_newton)
    # Check boundary conditions
    if G(0) < G(10**-15) and G(0) < 0:
        list_y.append(0)
    if G(1) < G(1 - 10**-15) and G(1) < 0:
        list_y.append(1)

    return list_y

# function of get the matrix G for inquality constraints
def C_matrix(y_list,num_mu, num_s, b):
    k=num_mu+num_s  
    def c_interior(num_mu, num_s, b, y):
        i = 1  # Fixed as per the input
        K = max(num_mu, num_s)
        k=num_mu+num_s
        check_c = np.zeros(num_mu)
        hat_c= np.zeros(num_s)
        
        for t in range(i, num_mu):
            term = sum(factorial(t) * (-0.5)**(t - i - u) / (factorial(t - i - u) * factorial(u)) * y**u * y* (1-y) for u in range(t - i + 1))
            check_c[t]= term
        if num_s>0:
            for t in range(0, num_s):
                term1 = log( y / (1 - y)) * sum(factorial(t) * (-0.5)**(t - i - u) / (factorial(t - i - u) * factorial(u)) * y**u * y* (1-y) for u in range(t - i + 1))
                term2 = sum(b[t,u,i]* y**u for u in range(t + i)) 
                hat_c[t] += term1
                hat_c[t] += term2
        else:
            pass
        c=np.zeros(k)

        # Indices to track positions in check_a and hat_a
        check_idx = 0
        hat_idx = 0

        # Iterate over the range of k and reconstruct a
        for i in range(k):
            if i % 4 == 0 or i % 4 == 3:  # Indices belonging to check_a
                c[i] = check_c[check_idx]
                check_idx += 1
            elif i % 4 == 1 or i % 4 == 2:  # Indices belonging to hat_a
                c[i] = hat_c[hat_idx]
                hat_idx += 1
        c=np.array(c)
        return c
    
    def c_corner(num_s, y):
        hat_c= np.zeros(num_s)
        if num_s>0:
            for t in range(0, num_s):
                hat_c[t] = (y-0.5)**t
        else:
            pass
        check_c = np.zeros(num_mu)
        c=np.zeros(k)
        # Indices to track positions in check_a and hat_a
        check_idx = 0
        hat_idx = 0

        # Iterate over the range of k and reconstruct a
        for i in range(k):
            if i % 4 == 0 or i % 4 == 3:  # Indices belonging to check_a
                c[i] = check_c[check_idx]
                check_idx += 1
            elif i % 4 == 1 or i % 4 == 2:  # Indices belonging to hat_a
                c[i] = hat_c[hat_idx]
                hat_idx += 1
        return c
    C=[]
    for y in y_list:
        if y>1 or y<0:
            raise ValueError("y is out of the bound [0,1].")
        elif 0<y<1:
            c=c_interior(num_mu, num_s, b, y)
        elif y==0 or y==1:
            c=c_corner(num_s, y)
        C.append(c)
    return C


def calculate_Y(y,k):
    # Number of rows in Y matches the number of y points
    n = len(y)
    # Initialize Y matrix
    Y = np.zeros((n, k))
    # Fill in Y matrix
    for j in range(k):
        power = j // 2
        if j % 4 == 0 or j % 4 == 3:  # Even-indexed columns
            Y[:, j] = (y - 0.5) ** power
        else:  # Odd-indexed columns
            Y[:, j] = (y - 0.5) ** power * np.log(y / (1 - y))
    return Y

def f(a, x, Y):
    residual = x - Y @ a
    return residual.T @ residual
    
# Algorithm to find the best a*
def find_a_star(k, x, y):
    start_time = time.time()  # Start timing the process
    tol=10e-6
    epsilon=10e-6
    # Calculate Y matrix and a_ols
    Y=calculate_Y(y,k)
    Y_T = Y.T
    YTY_inv = np.linalg.inv(Y_T @ Y)
    a_ols = YTY_inv @ Y_T @ x
    a_ols = tuple(a_ols)
    
    # Iterate over the coefficients and assign in "Z" ordering pattern
    check_a_ols = []
    hat_a_ols = []
    for i in range(k):
        if i % 4 == 0 or i % 4 == 3:  # Indices 0, 3, 4, 7, 8, ... -> check_a
            check_a_ols.append(a_ols[i])
        elif i % 4 == 1 or i % 4 == 2:  # Indices 1, 2, 5, 6, 9, 10, ... -> hat_a
            hat_a_ols.append(a_ols[i])
    
    # Number of mu and s should correspond to the number of check_a and hat_a
    num_mu = len(check_a_ols)
    num_s = len(hat_a_ols)
    K=max(num_mu, num_s)
    # Compute the b matrix
    b = compute_b_matrix(k, num_s)
    
    # Initialize the list of y values
    feasible = False
    y_list=[]
    a_list = []
    # Define the objective of the QP problem
    Q=matrix(2*Y.T @ Y)
    c=matrix(-2*Y.T @ x)
    
    iterations = 0  # Initialize the iteration counter
    while not feasible:
        if iterations == 0:
            a_current = a_ols
        else:
            # Compute the matrix G for inequality constraints
            G = C_matrix(y_list, num_mu, num_s, b)
            G= -1* np.array(G)
            G = matrix(G, tc='d')
            h = matrix(np.array([-epsilon] * len(y_list)))
            # Solve the QP problem
            a_current = solvers.qp(Q, c, G, h)['x']
            a_current = np.array(a_current).flatten()
        # Use grid to check the feasibility
        if K>2:
            y_list_i = grid_search_newtons_method(a_current, b, tol)
        elif K<=2:
            # Initialize a vector to hold the coefficients for y^0, y^1, ..., y^(j-1)
            check_a = []  # Coefficients assigned to mu
            hat_a = []    # Coefficients assigned to s            
            for j in range(k):
                if j % 4 == 0 or j % 4 == 3:  # Indices 0, 3, 4, 7, 8, ... -> check_a
                    check_a.append(a_current[j])
                elif j % 4 == 1 or j % 4 == 2:  # Indices 1, 2, 5, 6, 9, 10, ... -> hat_a
                    hat_a.append(a_current[j])
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
            y_list_i = []
            for root in roots:
                if G_value(a_current, b , root) < 0:
                    y_list_i.append(root)
            if len(y_list_i)>0 or y_list_i is not None:
                y_list_i = [root for root in y_list_i if root > tol and root< 1- tol]
            y_list_i.sort()
        if y_list_i == []:
            feasible_check = fb(a_current)
            if feasible_check[3]:
                feasible = True
            else:
                if not feasible_check[4]:
                    roots = feasible_check[1]
                    slopes = feasible_check[2]
                    for j in range(len(roots)):
                        if slopes[j] < 0:
                            y_list.append(roots[j])
                else:
                    if not feasible_check[-2]:
                        y_list.append(0)
                    if not feasible_check[-1]:
                        y_list.append(1)
        else:
            y_list.extend(y_list_i)
        iterations += 1  # Increment the iteration counter

    a_star = a_current
    f_a_star = f(a_star, x, Y)
    running_time = time.time() - start_time
    return {
        "a_ols": a_ols,
        "f(a_ols)": f(a_ols, x, Y),
        "Best a*": a_star,
        "f(a*)": f_a_star,
        "Iterations": iterations,
        "Running time": running_time
    }

# Define a function to process datasets for different values of k
def process_datasets(datasets, k_values, output_file):
    results = []

    for dataset_name, data in datasets.items():
        x = data['x']
        y = data['y']

        for k in k_values[dataset_name]:
            result = find_a_star(k, x, y)

            # Extract relevant details
            a_ols = result["a_ols"]  # Assuming a_ols is returned
            f_a_ols = result["f(a_ols)"]
            a_star = result["Best a*"]
            f_a_star = result["f(a*)"]
            iterations = result.get("Iterations", None)  # If iterations are returned
            running_time = result["Running time"]

            # Append results
            results.append({
                "Dataset": dataset_name,
                "k": k,
                "a_ols": round_list(a_ols),
                "f(a_ols)": round(f_a_ols,6),
                "a_star": round_list(a_star),
                "f(a*)": round(f_a_star,6),
                "Iterations": iterations,
                "Running Time (s)": round(running_time,6)
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save results to an Excel file
    results_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")
# Load the "5 data sets" sheet into a DataFrame to examine its structure
excel_data = pd.ExcelFile("/Users/Stephen/Dropbox/metalog_project/dataset/five data sets v1.xlsx")
five_datasets_df = excel_data.parse('5 data sets')


# Prepare datasets and k-values
k_values = {
    "Dataset_1": [6,10,16],
    "Dataset_2": [6,10,16],
    "Dataset_3": [6,10,16],
    "Dataset_4": [5, 6, 7],
    "Dataset_5": [4, 5, 6]
}

# Load the datasets from Excel
excel_data = pd.ExcelFile("/Users/Stephen/Dropbox/metalog_project/dataset/five data sets v1.xlsx")
five_datasets_df = excel_data.parse('5 data sets')

datasets = {}

for col in range(2, five_datasets_df.shape[1], 2):  # Loop through every second column for x and y pairs
    dataset_name = f"Dataset_{(col // 2)}"
    x_values = five_datasets_df.iloc[3:, col - 1].dropna().astype(float).values
    y_values = five_datasets_df.iloc[3:, col].dropna().astype(float).values
    datasets[dataset_name] = {'x': x_values, 'y': y_values}

# Process datasets and save results
output_file = "/Users/Stephen/Dropbox/metalog_project/dataset/results.xlsx"
process_datasets(datasets, k_values, output_file)
