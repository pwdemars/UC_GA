"""
The genetic_algorithm module includes two functions: 
    
1. run_genetic_algorithm(): this runs the GA, returning the best genotype in the 
final generation and a list of best fitnesses for each generation.
2. generate_offspring(): applies the operators in operators.py to create new
genotypes.
"""
import numpy as np

from helpers import convert_to_binary, convert_to_integer
from operators import *
from genotype import Genotype, Population

def run_genetic_algorithm(number_of_generations, seed_schedules, **kwargs):
    """
    Run the genetic algorithm. 
    
    Args:
        - seed schedules (array): this should be an array of dimension
        pop_size*T*num_gen, comprising pop_size random binary schedules. 
    """
    # Retrieve variables
    init_status = kwargs.get('init_status')
    pop_size = kwargs.get('pop_size')
    swap_window_hc_probability = kwargs.get('swap_window_hc_probability')    
    
    # Initialise parent population
    parent_population = Population(pop_size)
    # Population with seed_schedules (typically random)
    for i in range(pop_size):
        g = Genotype(convert_to_integer(seed_schedules[i], init_status), **kwargs)
        parent_population.add_genotype(g)
        
    # Get best genotype
    best_genotype1 = parent_population.best_two_genotypes()[0]

    # Initialise results list
    results = []
    
    for g in range(number_of_generations):
        
        # Initialise new population
        new_population = Population(pop_size)
        
        # Increase the value of the penalty
        new_penalty = kwargs.get('max_penalty')*(g+1)/number_of_generations
        kwargs.update({'constraint_penalty':new_penalty})
        
        # Always add the best genotype
        new_population.add_genotype(best_genotype1)
        
        while new_population.num_used < new_population.size:
            # Roulette wheel selection
            g1, g2 = parent_population.select_two_genotypes()       
            
            # Create a new offspring
            offspring1, offspring2 = generate_offspring(g1, g2, **kwargs)
            new_population.add_genotype(offspring1)
            new_population.add_genotype(offspring2)
            
        # Get best genotype and apply hill-climb operators
        # Also update best genotypes.
        best_genotype1, best_genotype2 = new_population.best_two_genotypes()
        new_population.remove_genotype(best_genotype1)
        
        # Apply swap mutation operator 
        best_binary_schedule = convert_to_binary(best_genotype1.schedule)
        best_binary_schedule = swap_mutation_operator(best_binary_schedule, **kwargs)
        
        # Apply swam window hill climb operator (with probability swap_window_hc_probability)
        random = np.random.uniform(0, 1)
        if random < swap_window_hc_probability:
            best_binary_schedule = swap_window_hc_operator(best_binary_schedule, **kwargs)
        
        # Convert back to integer
        best_integer_schedule = convert_to_integer(best_binary_schedule, init_status)
        best_genotype1 = Genotype(best_integer_schedule, **kwargs)
        
        # Return best schedule to new_population
        new_population.add_genotype(best_genotype1)
                
        # Make new_population the parent_population
        parent_population = new_population
        
        # Best fitness, add to results
        best_fitness = best_genotype1.fitness
        results.append(best_fitness)
        
        print("Best fitness at iteration {0}: {1}".format(g, best_fitness))
        
    return best_genotype1, results, new_population

def generate_offspring(genotype1, genotype2, **kwargs):
    """
    Use the crossover and mutation operators to produce an offspring from 2 genotypes.
    
    The mutation operator is always applied, but the probability that a given bit is mutated is 
    determined by mutation_probability. Similarly, the crossover operator is applied to each
    generator with probability crossover_probability.
    
    Args:
        - genotype1, genotype2 (Genotype objects): the genotypes to be combined
        
    Returns: 
        - offspring (Genotype object)
    """
    crossover_probability = kwargs.get('crossover_probability')
    mutation_probability = kwargs.get('mutation_probability')
    swap_window_probability = kwargs.get('swap_window_probability')
    window_mutation_probability = kwargs.get('window_mutation_probability')
    init_status = kwargs.get('init_status')
    
    binary_schedule1 = convert_to_binary(genotype1.schedule)
    binary_schedule2 = convert_to_binary(genotype2.schedule)
    
    # Apply crossover
    offspring_binary_schedule1, offspring_binary_schedule2 = crossover(binary_schedule1, binary_schedule2, 
                                                                       crossover_probability)
    
    # Apply mutation
    offspring_binary_schedule1 = mutate(offspring_binary_schedule1, mutation_probability)
    offspring_binary_schedule2 = mutate(offspring_binary_schedule2, mutation_probability)
    
    # Apply swap window operator
    r1, r2 = np.random.uniform(0, 1, size=2)
    if r1 < swap_window_probability:
        offspring_binary_schedule1 = swap_window(offspring_binary_schedule1)
    if r2 < swap_window_probability:
        offspring_binary_schedule2 = swap_window(offspring_binary_schedule2)
        
    # Apply window mutation operator
    r1, r2 = np.random.uniform(0, 1, size=2)
    if r1 < window_mutation_probability:
        offspring_binary_schedule1 = window_mutation(offspring_binary_schedule1)
    if r2 < window_mutation_probability:
        offspring_binary_schedule2 = window_mutation(offspring_binary_schedule2)
        
    # Convert to integer
    offspring_integer_schedule1 = convert_to_integer(offspring_binary_schedule1, init_status)
    offspring_integer_schedule2 = convert_to_integer(offspring_binary_schedule2, init_status)
    
    # Initialise genotype
    offspring1 = Genotype(offspring_integer_schedule1, **kwargs)
    offspring2 = Genotype(offspring_integer_schedule2, **kwargs)
    
    return offspring1, offspring2