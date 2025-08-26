"""
Genome representations for genetic algorithm optimization.
"""
from typing import List, Tuple
import numpy as np
from copy import deepcopy


class RealGenome:
    """
    Real-valued genome for continuous optimization.
    
    A genome represents a solution candidate as a 1D numpy array of real numbers.
    Each element in the array represents a parameter value to be optimized.
    """
    
    def __init__(
        self,
        num_parameters: int,
        parameter_bounds: List[float],
        hard_to_soft_mutation_ratio: float = 0.5,
        soft_mutation_range: Tuple[float, float] = (0.9, 1.1),
        genes: np.ndarray = None
    ):
        """
        Initialize a real-valued genome.
        
        Args:
            num_parameters: Number of parameters (genes) in the genome
            parameter_bounds: [min, max] bounds that apply to all parameters
            hard_to_soft_mutation_ratio: Probability of hard vs soft mutation
            soft_mutation_range: (min, max) multiplier range for soft mutations
            genes: Optional initial parameter values (if None, random initialization)
        """
        self.num_parameters = num_parameters
        self.parameter_bounds = parameter_bounds
        self.hard_to_soft_mutation_ratio = hard_to_soft_mutation_ratio
        self.soft_mutation_range = soft_mutation_range
        
        if genes is not None:
            self.genes = genes.copy()
        else:
            # Random initialization within bounds
            self.genes = np.random.uniform(
                parameter_bounds[0], 
                parameter_bounds[1], 
                size=num_parameters
            )
    
    def evaluate(self, fitness_function) -> float:
        """
        Evaluate this genome using the fitness function.
        
        Args:
            fitness_function: Function that takes 1D numpy array and returns fitness
            
        Returns:
            Fitness value
        """
        return fitness_function(self.genes)
    
    def hard_mutation(self) -> 'RealGenome':
        """
        Perform hard mutation: replace one random gene with new random value.
        
        Returns:
            New mutated genome
        """
        new_genome = deepcopy(self)
        random_index = np.random.randint(0, len(self.genes))
        new_genome.genes[random_index] = np.random.uniform(
            self.parameter_bounds[0], 
            self.parameter_bounds[1]
        )
        return new_genome
    
    def soft_mutation(self) -> 'RealGenome':
        """
        Perform soft mutation: multiply one random gene by random factor.
        
        Returns:
            New mutated genome
        """
        new_genome = deepcopy(self)
        random_index = np.random.randint(0, len(self.genes))
        multiplier = np.random.uniform(*self.soft_mutation_range)
        new_genome.genes[random_index] *= multiplier
        
        # Ensure the mutated gene stays within bounds
        new_genome.genes[random_index] = np.clip(
            new_genome.genes[random_index], 
            self.parameter_bounds[0], 
            self.parameter_bounds[1]
        )
        return new_genome
    
    def mutate(self) -> 'RealGenome':
        """
        Perform mutation (hard or soft based on ratio).
        
        Returns:
            New mutated genome
        """
        if np.random.uniform() < self.hard_to_soft_mutation_ratio:
            return self.hard_mutation()
        else:
            return self.soft_mutation()
    
    def crossover(self, other: 'RealGenome') -> Tuple['RealGenome', 'RealGenome']:
        """
        Perform weighted blend crossover with another genome.
        
        Args:
            other: Another RealGenome to crossover with
            
        Returns:
            Tuple of two offspring genomes
        """
        # Random blend factor
        beta = np.random.uniform()
        
        # Weighted blend crossover
        offspring1_genes = beta * self.genes + (1 - beta) * other.genes
        offspring2_genes = beta * other.genes + (1 - beta) * self.genes
        
        # Create offspring with same parameters as parents
        offspring1 = RealGenome(
            num_parameters=self.num_parameters,
            parameter_bounds=self.parameter_bounds,
            hard_to_soft_mutation_ratio=self.hard_to_soft_mutation_ratio,
            soft_mutation_range=self.soft_mutation_range,
            genes=offspring1_genes
        )
        
        offspring2 = RealGenome(
            num_parameters=self.num_parameters,
            parameter_bounds=self.parameter_bounds,
            hard_to_soft_mutation_ratio=self.hard_to_soft_mutation_ratio,
            soft_mutation_range=self.soft_mutation_range,
            genes=offspring2_genes
        )
        
        return offspring1, offspring2
    
    def __len__(self) -> int:
        """Return the number of parameters (genes)."""
        return len(self.genes)
    
    def __str__(self) -> str:
        """String representation of the genome."""
        return f"RealGenome({self.genes})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"RealGenome(genes={self.genes}, bounds={self.parameter_bounds})"
