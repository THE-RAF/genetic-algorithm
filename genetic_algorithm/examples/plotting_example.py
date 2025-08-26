from genetic_algorithm import optimize

def sphere_5d(x):
    return -sum(xi**2 for xi in x)  # 5D sphere function

result = optimize(
    fitness_function=sphere_5d,
    num_parameters=5,
    parameter_bounds=[-2, 2],
    population_size=25,
    generations=80,
    crossover_rate=0.9,
    plot_training_history=True
)

print(f"5D optimization with plotting complete!")
print(f"Best fitness: {result['best_fitness']:.4f}")
print(f"Solution norm: {(sum(x**2 for x in result['best_solution']))**0.5:.4f}")
