import random
from deap import base, creator, tools, algorithms

# Define the Rosenbrock function
def rosenbrock(x):
    a = 1
    b = 100
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

# Define a boundary function that can be used as an evaluation function
def boundary(fn, x):
    target = fn(x)
    def boundary_fn(xb):
        return (target - fn(xb))**2,
    return boundary_fn

# Define the problem as a minimization
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Define individual and population
def create_individual():
    return [random.uniform(-10, 10) for _ in range(2)]

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define evaluation function
evaluate = boundary(rosenbrock, [0, 0])

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Genetic Algorithm
population = toolbox.population(n=50)
ngen, cxpb, mutpb = 40, 0.5, 0.2
result = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)

# Select and print the top N individuals
N = 5
top_individuals = tools.selBest(population, k=N)
print(f"Top {N} individuals:")
for i, ind in enumerate(top_individuals, 1):
    print(f"Individual {i}: {ind}, Fitness: {evaluate(ind)[0]}")
