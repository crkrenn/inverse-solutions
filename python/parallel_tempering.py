import numpy as np

# Define the Rosenbrock function
def rosenbrock(x):
    a = 1
    b = 100
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def boundary(fn, x):
    target = fn(x)
    def boundary_fn(xb):
        return (target - fn(xb))**2
    return boundary_fn

objective_function = boundary(rosenbrock, [0.12345, 0.54321])


# Define a function to perform Metropolis criterion
def metropolis(delta, beta):
    if delta < 0:
        return True
    else:
        return np.random.rand() < np.exp(-beta * delta)

# Parallel Tempering class
class ParallelTempering:
    def __init__(self, func, bounds, n_replicas=10, max_iter=1000, swap_interval=10):
        self.func = func
        self.bounds = bounds
        self.n_replicas = n_replicas
        self.max_iter = max_iter
        self.swap_interval = swap_interval
        self.temperatures = np.logspace(0, 1, n_replicas)
        self.replicas = [np.random.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(n_replicas)]
        self.energies = np.array([func(replica) for replica in self.replicas])
        self.best_solutions = []

    def run(self):
        for iteration in range(self.max_iter):
            for i in range(self.n_replicas):
                current_energy = self.energies[i]
                proposed_solution = self.replicas[i] + np.random.normal(0, 1, self.replicas[i].shape)
                proposed_energy = self.func(proposed_solution)

                if metropolis(proposed_energy - current_energy, 1.0 / self.temperatures[i]):
                    self.replicas[i] = proposed_solution
                    self.energies[i] = proposed_energy

                # Update the best solutions found
                self.best_solutions.append((self.replicas[i], self.energies[i]))

            if iteration % self.swap_interval == 0:
                self.swap_replicas()

            # Debug information
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Current best energy: {min(self.energies)}")

        # Sort and return the best 5 solutions
        # self.best_solutions = sorted(self.best_solutions, key=lambda x: x[1])[:5]
        self.best_solutions = sorted(self.best_solutions, key=lambda x: x[1])
        return self.best_solutions

    def swap_replicas(self):
        for i in range(self.n_replicas - 1):
            if metropolis((self.energies[i+1] - self.energies[i]) * (1.0/self.temperatures[i] - 1.0/self.temperatures[i+1]), 1.0):
                self.replicas[i], self.replicas[i+1] = self.replicas[i+1], self.replicas[i]
                self.energies[i], self.energies[i+1] = self.energies[i+1], self.energies[i]

# Define bounds for the Rosenbrock function
# bounds = np.array([[-2, 2], [-1, 3]])
bounds = np.array([[-10, 10], [-10, 10]])

# Initialize and run Parallel Tempering
pt = ParallelTempering(objective_function, bounds, n_replicas=100, max_iter=1000, swap_interval=10)
best_solutions = pt.run()

# Print the best 5 solutions
print("\nTop 5 solutions:")
for i, (sol, val) in enumerate(best_solutions[:20], 1):
    print(f"Solution {i}: {sol}, Objective value: {val}")
