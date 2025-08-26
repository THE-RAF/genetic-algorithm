from genetic_algorithm import optimize

def simple_function(x):
    return -(x[0] - 2)**2  # Find x that minimizes (x-2)^2, so x=2

result = optimize(
    fitness_function=simple_function,
    num_parameters=1,
    parameter_bounds=[-5, 5],
    population_size=20,
    generations=50
)

print(f"Best solution: {result['best_solution'][0]:.3f}")
print(f"Target was: 2.000")
