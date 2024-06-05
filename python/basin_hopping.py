import numpy as np
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt

# Define the Rosenbrock function
def rosenbrock(x):
    a = 1
    b = 100
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

# Define a boundary function that can be used as an evaluation function
def boundary(fn, x):
    target = fn(x)
    # target = 0
    def boundary_fn(xb):
        return (target - fn(xb))**2
    return boundary_fn

# Create the objective function
objective_function = boundary(rosenbrock, [1.0, 0.99])

# Define a class to track local minima
class LocalMinimaTracker:
    def __init__(self):
        self.local_minima = []

    def __call__(self, x, f, accepted):
        if accepted:
            self.local_minima.append((x.copy(), f))

    def get_history(self):
        return self.local_minima

# Initialize the tracker
tracker = LocalMinimaTracker()

# Set the initial guess
x0 = [10, 10]
x0 = [0, 0]

# Perform the basin-hopping minimization
result = basinhopping(objective_function, x0, callback=tracker)

# Print the result
print("Global minimum found: x = {}, f(x) = {}".format(result.x, result.fun))

# Print the history of local minima
print("\nHistory of local minima found:")
for i, (x, f) in enumerate(tracker.get_history()):
    print("Local minimum {}: x = {}, f(x) = {}".format(i+1, x, f))

# Plot the local minima
history = tracker.get_history()
x_vals = [x[0] for x, _ in history]
y_vals = [x[1] for x, _ in history]
z_vals = [f for _, f in history]

plt.figure(figsize=(10, 6))
plt.scatter(x_vals, y_vals, c=z_vals, cmap='viridis', marker='o')
plt.colorbar(label='Objective function value')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('Local Minima Found During Basin-Hopping')
plt.grid(True)
plt.show()
