"""
Simple visualization utilities for genetic algorithm results.
"""


def plot_fitness_history(result_dict: dict, title: str = "Genetic Algorithm Fitness Evolution"):
    """
    Plot the evolution of best, average, and minimum fitness over generations.
    
    Args:
        result_dict: Dict returned from optimize() function containing fitness history
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt

        if 'seaborn-v0_8' in plt.style.available:
            plt.style.use('seaborn-v0_8')

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the three essential fitness metrics
        ax.plot(result_dict['best_fitnesses'], color='#2E8B57', label='Best Fitness', linewidth=2)
        ax.plot(result_dict['mean_fitnesses'], color='#4169E1', label='Average Fitness', linewidth=2)
        ax.plot(result_dict['min_fitnesses'], color='#DC143C', label='Minimum Fitness', linewidth=2)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Cannot plot fitness evolution.")
        print(f"Best fitness progression: {result_dict['best_fitnesses']}")
        print(f"Final best fitness: {result_dict['best_fitness']}")
