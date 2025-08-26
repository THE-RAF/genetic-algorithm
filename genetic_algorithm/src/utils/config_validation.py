"""
Configuration validation and parameter handling for the genetic algorithm.

This module provides comprehensive validation of all genetic algorithm parameters
to ensure they are valid, consistent, and within acceptable ranges.
"""
from typing import List, Tuple, Dict, Any
import numpy as np


def validate_parameters(
    num_parameters: int,
    parameter_bounds: List[float],
    population_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    selection_method: str,
    tournament_k: int,
    elitism_number: int,
    hard_to_soft_mutation_ratio: float,
    soft_mutation_range: Tuple[float, float]
) -> Dict[str, Any]:
    """
    Validate genetic algorithm parameters.
    
    Returns:
        Dict containing validated parameters
    
    Raises:
        ValueError: If any parameters are invalid
    """
    # Validate num_parameters
    if not isinstance(num_parameters, int) or num_parameters < 1:
        raise ValueError("num_parameters must be a positive integer")
    
    # Validate parameter_bounds
    if not isinstance(parameter_bounds, (list, tuple)) or len(parameter_bounds) != 2:
        raise ValueError("parameter_bounds must be [min, max]")
    if not (isinstance(parameter_bounds[0], (int, float)) and isinstance(parameter_bounds[1], (int, float))):
        raise ValueError("parameter_bounds must contain numeric values")
    if parameter_bounds[0] >= parameter_bounds[1]:
        raise ValueError("parameter_bounds: min must be less than max")
    
    bounds = parameter_bounds
    
    # Validate population_size
    if not isinstance(population_size, int) or population_size < 2:
        raise ValueError("population_size must be an integer >= 2")
    
    # Validate generations
    if not isinstance(generations, int) or generations < 1:
        raise ValueError("generations must be a positive integer")
    
    # Validate crossover_rate
    if not isinstance(crossover_rate, (int, float)) or not (0.0 <= crossover_rate <= 1.0):
        raise ValueError("crossover_rate must be a float between 0.0 and 1.0")
    
    # Validate mutation_rate
    if not isinstance(mutation_rate, (int, float)) or not (0.0 <= mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be a float between 0.0 and 1.0")
    
    # Validate selection_method
    if selection_method not in ['tournament', 'roulette']:
        raise ValueError("selection_method must be 'tournament' or 'roulette'")
    
    # Validate tournament_k
    if not isinstance(tournament_k, int) or tournament_k < 1:
        raise ValueError("tournament_k must be a positive integer")
    if tournament_k > population_size:
        raise ValueError("tournament_k cannot be larger than population_size")
    
    # Validate elitism_number
    if not isinstance(elitism_number, int) or elitism_number < 0:
        raise ValueError("elitism_number must be a non-negative integer")
    if elitism_number >= population_size:
        raise ValueError("elitism_number must be less than population_size")
    
    # Validate hard_to_soft_mutation_ratio
    if not isinstance(hard_to_soft_mutation_ratio, (int, float)) or not (0.0 <= hard_to_soft_mutation_ratio <= 1.0):
        raise ValueError("hard_to_soft_mutation_ratio must be a float between 0.0 and 1.0")
    
    # Validate soft_mutation_range
    if not isinstance(soft_mutation_range, (list, tuple)) or len(soft_mutation_range) != 2:
        raise ValueError("soft_mutation_range must be (min, max) tuple")
    if soft_mutation_range[0] >= soft_mutation_range[1]:
        raise ValueError("soft_mutation_range: min must be less than max")
    if soft_mutation_range[0] <= 0:
        raise ValueError("soft_mutation_range values must be positive")
    
    return {
        'num_parameters': num_parameters,
        'parameter_bounds': bounds,
        'population_size': population_size,
        'generations': generations,
        'crossover_rate': float(crossover_rate),
        'mutation_rate': float(mutation_rate),
        'selection_method': selection_method,
        'tournament_k': tournament_k,
        'elitism_number': elitism_number,
        'hard_to_soft_mutation_ratio': float(hard_to_soft_mutation_ratio),
        'soft_mutation_range': tuple(soft_mutation_range)
    }
