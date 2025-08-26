from genetic_algorithm import optimize

def peaks(x):
    return -(x[0]*x[1]*x[2])  # Find maximum at corners

result = optimize(
    fitness_function=peaks,
    num_parameters=3,
    parameter_bounds=[-1, 1],
    population_size=40,
    generations=60,
    mutation_rate=0.3,
    hard_to_soft_mutation_ratio=0.7,
    soft_mutation_range=(0.8, 1.2)
)

print(f"Custom mutation result: ({result['best_solution'][0]:.3f}, {result['best_solution'][1]:.3f}, {result['best_solution'][2]:.3f})")
print(f"Best fitness: {result['best_fitness']:.3f}")
