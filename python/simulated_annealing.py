import numpy as np
from scipy.optimize import dual_annealing

# Define the Rosenbrock function
def rosenbrock(x):
    a = 1
    b = 100
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

# Define a boundary function that can be used as an evaluation function
def boundary(fn, x):
    target = fn(x)
    def boundary_fn(xb):
        return (target - fn(xb))**2
    return boundary_fn

objective_function = boundary(rosenbrock, [0.12345, 0.54321])

# To store intermediate steps
intermediate_solutions = []

# Define callback function to capture intermediate solutions
def callback(x, f, context):
    intermediate_solutions.append((x, f))
    print(f"Intermediate solution: {x}, Objective value: {f}")

bounds = [(-10, 10), (-10, 10)]
result = dual_annealing(objective_function, bounds, callback=callback)
print("Best solution is:", result.x)
print("Best objective value is:", result.fun)

# Sort and print the best 5 solutions
sorted_solutions = sorted(intermediate_solutions, key=lambda x: x[1])
print("\nTop 5 solutions:")
for i, (sol, val) in enumerate(sorted_solutions[:5], 1):
    print(f"Solution {i}: {sol}, Objective value: {val}")
