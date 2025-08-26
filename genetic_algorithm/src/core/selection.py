"""
Selection strategies for genetic algorithm.
"""
from typing import List
import numpy as np
import random
from copy import deepcopy


class Selection:
    """
    Static methods for different selection strategies in genetic algorithms.
    """
    
    @staticmethod
    def roulette(population: List, fitnesses: List[float]) -> List:
        """
        Roulette wheel selection based on fitness proportions.
        
        Args:
            population: List of genome objects
            fitnesses: List of fitness values for each genome
            
        Returns:
            List of selected genomes (same size as population)
        """
        # Normalize fitness values to positive weights
        # Shift all values to be positive, then use as weights
        fitness_array = np.array(fitnesses)
        min_fitness = min(fitness_array)
        
        # Shift to make all values positive (add small epsilon to avoid zeros)
        weights = fitness_array - min_fitness + 1e-10
        
        # Select genomes based on positive weights
        selected_genomes = random.choices(
            population, 
            weights=weights, 
            k=len(population)
        )
        
        return selected_genomes
    
    @staticmethod
    def tournament(population: List, fitnesses: List[float], k: int) -> List:
        """
        Tournament selection with k-way tournaments.
        
        Args:
            population: List of genome objects  
            fitnesses: List of fitness values for each genome
            k: Tournament size
            
        Returns:
            List of selected genomes (same size as population)
        """
        selected_genomes = []
        
        for _ in range(len(population)):
            # Randomly select k individuals for tournament
            tournament_indices = [random.randint(0, len(population) - 1) for _ in range(k)]
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            tournament_genomes = [population[i] for i in tournament_indices]
            
            # Select the genome with highest fitness
            winner_index = np.argmax(tournament_fitnesses)
            selected_genomes.append(tournament_genomes[winner_index])
        
        return selected_genomes
    
    @staticmethod
    def elitism(population: List, fitnesses: List[float], elitism_number: int) -> List:
        """
        Select the best individuals for elitism.
        
        Args:
            population: List of genome objects
            fitnesses: List of fitness values for each genome  
            elitism_number: Number of best individuals to select
            
        Returns:
            List of the best genomes (deep copies)
        """
        # Create indices and sort by fitness (descending)
        sorted_indices = list(range(len(fitnesses)))
        sorted_indices.sort(key=fitnesses.__getitem__, reverse=True)
        
        # Select the best genomes
        best_genomes = [deepcopy(population[i]) for i in sorted_indices[:elitism_number]]
        
        return best_genomes
