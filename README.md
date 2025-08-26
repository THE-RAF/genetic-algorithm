# Genetic Algorithm for Real-Valued Optimization

A clean, professional genetic algorithm implementation for optimizing real-valued parameter vectors.

## Features

- **Simple API**: Single `optimize()` function with sensible defaults
- **Flexible Parameters**: Configure population size, generations, mutation rates, and more
- **Multiple Selection Methods**: Tournament and roulette wheel selection
- **Dual Mutation Types**: Hard mutations (random replacement) and soft mutations (multiplicative adjustment)
- **Built-in Visualization**: Automatic fitness tracking and plotting
- **Type Hints**: Full type annotation support for modern Python development

## Quick Start

```python
from genetic_algorithm import optimize

def fitness_function(x):
    # Minimize sum of squares (Sphere function)
    return -(sum(xi**2 for xi in x))

# Run optimization
result = optimize(
    fitness_function=fitness_function,
    num_parameters=10,
    parameter_bounds=[-5, 5],
    population_size=100,
    generations=200,
    plot_training_history=True  # Shows fitness evolution plot
)

print(f"Best solution: {result['best_solution']}")
print(f"Best fitness: {result['best_fitness']}")
```

## Installation

Run in your terminal:

```bash
pip install git+https://github.com/THE-RAF/genetic-algorithm.git
```

## API Reference

### optimize()

Main optimization function with the following parameters:

- `fitness_function`: Function to optimize (takes 1D numpy array, returns float)
- `num_parameters`: Number of parameters to optimize
- `parameter_bounds`: `[min, max]` bounds that apply to all parameters
- `population_size`: Size of population (default: 100)
- `generations`: Number of generations to run (default: 100)
- `crossover_rate`: Probability of crossover (default: 0.8)
- `mutation_rate`: Probability of mutation (default: 0.1)
- `selection_method`: `'tournament'` or `'roulette'` (default: 'tournament')
- `tournament_k`: Tournament size for tournament selection (default: 3)
- `elitism_number`: Number of best individuals to preserve (default: 1)
- `hard_to_soft_mutation_ratio`: Ratio of hard vs soft mutations (default: 0.5)
- `soft_mutation_range`: `(min, max)` multiplier range for soft mutations (default: (0.9, 1.1))
- `plot_training_history`: Whether to automatically show fitness evolution plot (default: False)

### Return Value

Returns a dictionary with:
- `'best_solution'`: Best parameter vector found (1D numpy array)
- `'best_fitness'`: Best fitness value achieved
- `'best_fitnesses'`: List of best fitness per generation
- `'mean_fitnesses'`: List of mean fitness per generation  
- `'min_fitnesses'`: List of minimum fitness per generation
- `'generations'`: Number of generations executed

## Examples

The `examples/` directory contains focused examples showing different features:

- `simple_example.py`: Basic optimization with default settings
- `roulette_example.py`: Using roulette wheel selection
- `elitism_example.py`: High elitism for preserving good solutions
- `mutation_example.py`: Custom mutation parameters
- `plotting_example.py`: Manual control over fitness visualization

Run any example with:
```bash
python -m genetic_algorithm.examples.simple_example
```

## Algorithm Details

This genetic algorithm implements:

- **Real-valued genomes**: Each individual is a 1D numpy array of parameters
- **Weighted blend crossover**: Offspring = β × parent1 + (1-β) × parent2
- **Dual mutation strategy**: 
  - Hard mutations: Replace parameter with new random value
  - Soft mutations: Multiply parameter by random factor
- **Tournament selection**: Select best from random tournament
- **Roulette selection**: Probability proportional to fitness (handles negative values)
- **Elitism**: Preserve best individuals across generations

## Algorithm Pseudocode

Here's a high-level overview of the genetic algorithm implementation:

```
FUNCTION optimize(fitness_function, num_parameters, parameter_bounds, ...):
    1. VALIDATE all input parameters (bounds, rates, sizes, etc.)
    
    2. CREATE genome template:
       - RealGenome with parameter_bounds, mutation settings
       - Each genome = 1D numpy array of real numbers
    
    3. INITIALIZE population:
       - Create population_size random genomes
       - Each gene ∈ [parameter_bounds[0], parameter_bounds[1]]
    
    4. FOR each generation (1 to generations):
       a) EVALUATE FITNESS:
          - fitness[i] = fitness_function(genome[i].genes)
          - Track best, mean, minimum fitness
       
       b) ELITISM:
          - Select top elitism_number individuals
          - Deep copy them for preservation
       
       c) SELECTION:
          - Tournament: k random individuals → pick best
          - Roulette: probability ∝ normalized fitness
          - Generate parent population
       
       d) CROSSOVER (with crossover_rate probability):
          - Pair parents randomly
          - Weighted blend: child1 = β×parent1 + (1-β)×parent2
          -                child2 = β×parent2 + (1-β)×parent1
          - Where β ~ Uniform(0,1)
       
       e) MUTATION (with mutation_rate probability):
          - Hard mutation (hard_to_soft_mutation_ratio chance):
            * Replace random gene with new random value ∈ bounds
          - Soft mutation (otherwise):
            * Multiply random gene by factor ∈ soft_mutation_range
            * Clip result to bounds
       
       f) POPULATION REPLACEMENT:
          - Replace first N individuals with elite
          - Shuffle population randomly
       
       g) UPDATE STATISTICS:
          - Record best_fitness, mean_fitness, min_fitness
          - Update progress bar
    
    5. RETURN results dictionary:
       - best_solution: numpy array of optimal parameters
       - best_fitness: optimal fitness value
       - fitness histories: best_fitnesses, mean_fitnesses, min_fitnesses
       - generations: number of generations executed

CROSSOVER DETAILS:
    INPUT: parent1, parent2 (both RealGenomes)
    β = random(0, 1)
    child1.genes = β * parent1.genes + (1-β) * parent2.genes  
    child2.genes = β * parent2.genes + (1-β) * parent1.genes
    OUTPUT: child1, child2

SELECTION DETAILS:
    Tournament(population, fitnesses, k):
        FOR each selection:
            indices = k random indices
            winner = argmax(fitnesses[indices])
            SELECT population[winner]
    
    Roulette(population, fitnesses):
        weights = fitnesses - min(fitnesses) + ε  # Make positive
        SELECT using weighted random choice

MUTATION DETAILS:
    Hard mutation: genes[random_index] = uniform(bounds[0], bounds[1])
    Soft mutation: genes[random_index] *= uniform(soft_range[0], soft_range[1])
                   genes[random_index] = clip(genes[random_index], bounds)
```

## License

This genetic algorithm module is provided as-is for educational and research purposes.
