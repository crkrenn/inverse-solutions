import numpy as np
from pyswarm.pso import _initial_swarm, _update_velocity, _update_position, _evaluate_particles

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

objective_function = boundary(rosenbrock, [0, 0])

# To store intermediate steps
intermediate_solutions = []

# Define callback function to capture intermediate solutions
def callback(x, f):
    intermediate_solutions.append((x, f))
    print(f"Intermediate solution: {x}, Objective value: {f}")

# Custom PSO implementation with callback
def custom_pso(func, lb, ub, swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-8, minfunc=1e-8, debug=False, callback=None):
    lb = np.array(lb)
    ub = np.array(ub)
    vhigh = np.abs(ub - lb)
    vlow = -vhigh

    # Initialize the swarm
    swarm = _initial_swarm(lb, ub, vlow, vhigh, swarmsize)
    gbest = np.inf
    gbest_position = None

    for i in range(maxiter):
        # Evaluate particles
        p = swarm[0]
        x = p[0]
        v = p[1]
        fx = func(x)
        p[2] = fx

        # Update personal best
        pbest_mask = fx < p[3]
        p[3][pbest_mask] = fx[pbest_mask]
        p[4][pbest_mask] = x[pbest_mask]

        # Update global best
        if np.min(fx) < gbest:
            gbest = np.min(fx)
            gbest_position = x[np.argmin(fx)]

        # Update velocities and positions
        v = _update_velocity(x, v, p[4], gbest_position, vlow, vhigh, omega, phip, phig)
        x = _update_position(x, v, lb, ub)

        swarm[0] = (x, v, fx, p[3], p[4])

        if callback:
            for xi, fi in zip(x, fx):
                callback(xi, fi)

        if debug:
            print(f'Iteration {i+1}/{maxiter}, Best: {gbest}')

        # Check stopping criteria
        if np.abs(gbest) <= minfunc:
            break

    return gbest_position, gbest

# Define bounds
lb = [-10, -10]
ub = [10, 10]

# Run custom PSO with the custom callback
xopt, fopt = custom_pso(objective_function, lb, ub, callback=callback)
print("Best solution is:", xopt)
print("Best objective value is:", fopt)

# Sort and print the best 5 solutions
sorted_solutions = sorted(intermediate_solutions, key=lambda x: x[1])
print("\nTop 5 solutions:")
for i, (sol, val) in enumerate(sorted_solutions[:5], 1):
    print(f"Solution {i}: {sol}, Objective value: {val}")
