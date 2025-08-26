from genetic_algorithm import optimize

def quadratic(x):
    return -(x[0]**2 + x[1]**2)  # Find minimum at (0,0)

result = optimize(
    fitness_function=quadratic,
    num_parameters=2,
    parameter_bounds=[-3, 3],
    population_size=30,
    generations=40,
    selection_method='roulette'
)

print(f"Roulette selection result: ({result['best_solution'][0]:.3f}, {result['best_solution'][1]:.3f})")
print(f"Target: (0.000, 0.000)")
