"""
Genetic Algorithm for Real-Valued Optimization

A clean, professional genetic algorithm implementation for optimizing
real-valued parameter vectors.

Example usage:
    from genetic_algorithm import optimize, plot_fitness_history
    
    def fitness_function(x):
        return -(sum(xi**2 for xi in x))  # Minimize sum of squares
    
    # Basic usage
    result = optimize(
        fitness_function=fitness_function,
        num_parameters=10,
        parameter_bounds=[-5, 5],
        population_size=100,
        generations=200
    )
    
    # With automatic plotting
    result = optimize(
        fitness_function=fitness_function,
        num_parameters=10,
        parameter_bounds=[-5, 5],
        population_size=100,
        generations=200,
        plot_training_history=True  # Automatically shows plot
    )
    
    print(f"Best solution: {result['best_solution']}")
    print(f"Best fitness: {result['best_fitness']}")
    
    # Manual plotting for advanced control
    plot_fitness_history(result)
"""

from typing import List
from .src.core.genetic_algorithm import GeneticAlgorithm
from .src.core.genomes import RealGenome
from .src.utils.config_validation import validate_parameters
from .src.utils.visualization import plot_fitness_history

def optimize(
    fitness_function,
    num_parameters: int,
    parameter_bounds: List[float],
    population_size: int = 100,
    generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    selection_method: str = 'tournament',
    tournament_k: int = 3,
    elitism_number: int = 1,
    hard_to_soft_mutation_ratio: float = 0.5,
    soft_mutation_range = (0.9, 1.1),
    plot_training_history: bool = False
) -> dict:
    """
    Optimize a real-valued function using genetic algorithm.
    
    Args:
        fitness_function: Function to optimize, takes 1D numpy array, returns float
        num_parameters: Number of parameters to optimize
        parameter_bounds: [min, max] bounds that apply to all parameters
        population_size: Size of population
        generations: Number of generations to run
        crossover_rate: Probability of crossover (0.0 to 1.0)
        mutation_rate: Probability of mutation (0.0 to 1.0)
        selection_method: 'tournament' or 'roulette'
        tournament_k: Tournament size for tournament selection
        elitism_number: Number of best individuals to preserve
        hard_to_soft_mutation_ratio: Ratio of hard vs soft mutations (0.0 to 1.0)
        soft_mutation_range: (min, max) multiplier range for soft mutations
        plot_training_history: Whether to automatically plot fitness evolution after optimization
    
    Returns:
        Dict: Optimization results with keys:
            - 'best_solution': 1D numpy array of best parameter vector
            - 'best_fitness': Best fitness value achieved
            - 'best_fitnesses': List of best fitness per generation  
            - 'mean_fitnesses': List of mean fitness per generation
            - 'min_fitnesses': List of minimum fitness per generation
            - 'generations': Number of generations executed
    """
    # Validate all parameters
    config = validate_parameters(
        num_parameters=num_parameters,
        parameter_bounds=parameter_bounds,
        population_size=population_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        selection_method=selection_method,
        tournament_k=tournament_k,
        elitism_number=elitism_number,
        hard_to_soft_mutation_ratio=hard_to_soft_mutation_ratio,
        soft_mutation_range=soft_mutation_range
    )
    
    # Create genome specification
    genome = RealGenome(
        num_parameters=config['num_parameters'],
        parameter_bounds=config['parameter_bounds'],
        hard_to_soft_mutation_ratio=config['hard_to_soft_mutation_ratio'],
        soft_mutation_range=config['soft_mutation_range']
    )
    
    # Create and run genetic algorithm
    ga = GeneticAlgorithm(
        genome_class=genome,
        fitness_function=fitness_function,
        population_size=config['population_size'],
        generations=config['generations'],
        crossover_rate=config['crossover_rate'],
        mutation_rate=config['mutation_rate'],
        selection_method=config['selection_method'],
        tournament_k=config['tournament_k'],
        elitism_number=config['elitism_number']
    )
    
    result = ga.run()
    
    # Plot training history if requested
    if plot_training_history:
        plot_fitness_history(result, title="Training History")
    
    return result

__all__ = ['optimize', 'plot_fitness_history', 'GeneticAlgorithm', 'RealGenome']
