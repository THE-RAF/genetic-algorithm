from genetic_algorithm import optimize

def rosenbrock(x):
    return -(100*(x[1] - x[0]**2)**2 + (1 - x[0])**2)  # Find minimum at (1,1)

result = optimize(
    fitness_function=rosenbrock,
    num_parameters=2,
    parameter_bounds=[-2, 2],
    population_size=500,
    generations=100,
    elitism_number=10,
    tournament_k=5
)

print(f"High elitism result: ({result['best_solution'][0]:.3f}, {result['best_solution'][1]:.3f})")
print(f"Target: (1.000, 1.000)")
