"""
Main genetic algorithm implementation for real-valued optimization.
"""
from typing import List
import numpy as np
import random
from copy import deepcopy
from tqdm import tqdm

from .selection import Selection
from .genomes import RealGenome


class GeneticAlgorithm:
    """
    Genetic Algorithm for real-valued parameter optimization.
    
    Evolves a population of candidate solutions through selection, crossover,
    and mutation operations to find optimal parameter values.
    """
    
    def __init__(
        self,
        genome_class: RealGenome,
        fitness_function,
        population_size: int,
        generations: int,
        crossover_rate: float,
        mutation_rate: float,
        selection_method: str,
        tournament_k: int,
        elitism_number: int
    ):
        """
        Initialize the genetic algorithm.
        
        Args:
            genome_class: RealGenome instance used as template for population
            fitness_function: Function to optimize (takes 1D array, returns float)
            population_size: Size of the population
            generations: Number of generations to run
            crossover_rate: Probability of crossover (0.0 to 1.0)
            mutation_rate: Probability of mutation (0.0 to 1.0)  
            selection_method: 'tournament' or 'roulette'
            tournament_k: Tournament size for tournament selection
            elitism_number: Number of best individuals to preserve
        """
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_generations = generations
        self.pop_size = population_size
        self.elitism_number = elitism_number
        self.tournament_k = tournament_k
        self.selection_method = selection_method
        
        self.genome_template = genome_class
        self.fitness_function = fitness_function
        
        # Initialize population
        self.population = self._create_initial_population()
        
        # Tracking variables
        self.best_fitness = -np.inf
        self.best_genome = None
        self.best_fitnesses = []
        self.mean_fitnesses = []
        self.min_fitnesses = []
        
        # Generation tracking
        self.current_generation = 1
        self.done = False
    
    def _create_initial_population(self) -> List[RealGenome]:
        """Create initial population of genomes."""
        return [
            RealGenome(
                num_parameters=self.genome_template.num_parameters,
                parameter_bounds=self.genome_template.parameter_bounds,
                hard_to_soft_mutation_ratio=self.genome_template.hard_to_soft_mutation_ratio,
                soft_mutation_range=self.genome_template.soft_mutation_range
            )
            for _ in range(self.pop_size)
        ]
    
    def select(self, population: List[RealGenome], fitnesses: List[float], population_fitness: float) -> List[RealGenome]:
        """
        Select parents for next generation.
        
        Args:
            population: Current population
            fitnesses: Fitness values for each individual
            population_fitness: Sum of all fitness values
            
        Returns:
            Selected individuals for reproduction
        """
        if self.selection_method == 'roulette':
            return Selection.roulette(population, fitnesses, population_fitness)
        elif self.selection_method == 'tournament':
            return Selection.tournament(population, fitnesses, self.tournament_k)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def population_crossover(self, population: List[RealGenome]) -> List[RealGenome]:
        """
        Perform crossover on the population.
        
        Args:
            population: Population to perform crossover on
            
        Returns:
            Population after crossover
        """
        # Split population in half and pair them
        mid_point = len(population) // 2
        parents_half1 = population[:mid_point]
        parents_half2 = population[mid_point:]
        
        offspring = []
        for parent1, parent2 in zip(parents_half1, parents_half2):
            if np.random.uniform() < self.crossover_rate:
                child1, child2 = parent1.crossover(parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])
        
        return offspring
    
    def mutate_population(self, population: List[RealGenome]) -> List[RealGenome]:
        """
        Perform mutation on the population.
        
        Args:
            population: Population to mutate
            
        Returns:
            Population after mutation
        """
        mutated_population = []
        for genome in population:
            if np.random.uniform() < self.mutation_rate:
                mutated_population.append(genome.mutate())
            else:
                mutated_population.append(genome)
        
        return mutated_population
    
    def run_generation(self) -> tuple:
        """
        Run a single generation of the genetic algorithm.
        
        Returns:
            Tuple of (new_population, done_flag)
        """
        # Evaluate fitness for all individuals
        fitnesses = [genome.evaluate(self.fitness_function) for genome in self.population]
        population_fitness = sum(fitnesses)
        
        # Update best fitness and genome
        best_fitness_from_population = max(fitnesses)
        if best_fitness_from_population > self.best_fitness:
            self.best_fitness = best_fitness_from_population
            self.best_genome = deepcopy(self.population[np.argmax(fitnesses)])
        
        # Apply elitism - select best individuals
        elite_population = Selection.elitism(self.population, fitnesses, self.elitism_number)
        
        # Selection, crossover, and mutation
        self.population = self.select(self.population, fitnesses, population_fitness)
        self.population = self.population_crossover(self.population)
        self.population = self.mutate_population(self.population)
        
        # Replace first N individuals with elite, then shuffle
        self.population = elite_population + self.population[self.elitism_number:]
        random.shuffle(self.population)
        
        # Record fitness statistics
        self.best_fitnesses.append(self.best_fitness)
        self.mean_fitnesses.append(population_fitness / self.pop_size)
        self.min_fitnesses.append(min(fitnesses))
        
        # Update generation counter
        self.current_generation += 1
        self.done = self.current_generation >= self.num_generations
        
        return deepcopy(self.population), self.done
    
    def run(self) -> dict:
        """
        Run the complete genetic algorithm.
        
        Returns:
            Dict containing optimization results with keys:
            - 'best_solution': 1D numpy array of best parameter vector
            - 'best_fitness': Best fitness value achieved  
            - 'best_fitnesses': List of best fitness per generation
            - 'mean_fitnesses': List of mean fitness per generation
            - 'min_fitnesses': List of minimum fitness per generation
            - 'generations': Number of generations executed
        """
        with tqdm(total=self.num_generations, desc="GA Progress") as progress_bar:
            while not self.done:
                population, self.done = self.run_generation()
                progress_bar.update(1)
            
            # Final update to show 100% completion
            progress_bar.update(1)
        
        # Return results as simple dict
        return {
            'best_solution': self.best_genome.genes,
            'best_fitness': self.best_fitness,
            'best_fitnesses': self.best_fitnesses,
            'mean_fitnesses': self.mean_fitnesses,
            'min_fitnesses': self.min_fitnesses,
            'generations': self.current_generation - 1
        }
