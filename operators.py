#!/usr/bin/env python3

"""
The operators module includes all the operators used to create variation and 
improvement in a population of genotypes. There are 6 operators in this module

1. Crossover 
2. Mutation
3. Swap window
4. Window mutation
5. Swap mutation
6. Swap window hill-climb
"""
import numpy as np

from helpers import *
from fitness import calculate_schedule_fitness

def mutate(binary_schedule, mutation_probability):
    """
    Randomly change bits in a binary schedule.
    
    Each bit has a small probability of being switched.
    
    Args:
        - binary_schedule (array): a T x N array giving the on/off statuses 
        of generators
        - mutation_probability (float): probability of mutation (per bit)
        
    Returns:
        - mutated_binary_schedule (array): the altered binary schedule
    """
    T, num_gen = binary_schedule.shape
    random_matrix = np.random.uniform(0, 1, size=(T, num_gen))
    # Get boolean array for where mutations will occur 
    idx = np.where(random_matrix < mutation_probability)
    
    # Change bits according to index.
    binary_schedule[idx] = np.array([not x for x in binary_schedule[idx]])
    
    mutated_binary_schedule = binary_schedule

    return mutated_binary_schedule


def crossover(binary_schedule1, binary_schedule2, crossover_probability):
    """
    Crossover operator for the genetic algorithm.
    
    Randomly select a crossover point, and for each generator use schedule 1
    up to the crossover point, and schedule 2 past the crossover point.
    
    Args:
        - binary_schedule1, binary_schedule2 (array): binary schedules for the 
        two genotypes being combined.
        
    Returns:
        - offspring_binary_schedule (array): the new schedule
    """
    T, num_gen = binary_schedule1.shape
    
    # Choose a random crossover point
    crossover_point = np.random.choice(T)
    
    # Initialise the offspring binary schedules
    offspring_binary_schedule1 = np.copy(binary_schedule1)
    offspring_binary_schedule2 = np.copy(binary_schedule2)
    
    for n in range(num_gen):
        random = np.random.uniform(0, 1)
        if random < crossover_probability:
            offspring_binary_schedule1[0:crossover_point,n] = binary_schedule2[0:crossover_point,n]
            offspring_binary_schedule2[crossover_point:,n] = binary_schedule1[crossover_point:,n]

    return offspring_binary_schedule1, offspring_binary_schedule2

def swap_window(binary_schedule):
    """
    The swap window operator.
    
    Takes a random time window in the schedule and 2 random units and swaps
    the schedules of the 2 units between these points. 
    
    Args:
        - binary_schedule (array): the schedule to which the operator 
        will be applied.
        
    Returns:
        - new_binary_schedule (array): the altered binary schedule.
    """
    T, num_gen = binary_schedule.shape
    
    # Choose a time window
    random_window = np.random.choice(np.arange(T), size=2, replace=False)
    random_window.sort()
    
    # Choose 2 generators
    random_gens = np.random.choice(np.arange(num_gen), size=2, replace=False)
        
    # Copy the schedule
    new_binary_schedule = np.copy(binary_schedule)
    
    # Copy the first schedule to swap
    temp = np.copy(binary_schedule[random_window[0]:random_window[1],random_gens[0]])
    
    new_binary_schedule[random_window[0]:random_window[1],random_gens[0]] = new_binary_schedule[random_window[0]:random_window[1],random_gens[1]]
    new_binary_schedule[random_window[0]:random_window[1],random_gens[1]] = temp
    
    return new_binary_schedule

def window_mutation(binary_schedule):
    """
    Window mutation operator.
    
    Takes a random unit and time window and a binary schedule and changes
    all bits to either 0 or 1 (randomly). 
    
    Args:
        - binary_schedule (array): the schedule to which the operator 
        will be applied.
        
    Returns:
        - new_binary_schedule (array): the altered binary schedule.
    """
    T, num_gen = binary_schedule.shape
    
    # Choose a time window
    random_window = np.random.choice(np.arange(T), size=2, replace=False)
    random_window.sort()
    
    # Choose generator
    random_gen = np.random.choice(np.arange(num_gen))
    
    # Choose 0 or 1
    zero_or_one = np.random.choice(np.arange(2))
    
    new_binary_schedule = binary_schedule
    
    # Change to 0 or 1 in random window for random generator
    new_binary_schedule[random_window[0]:random_window[1],random_gen] = zero_or_one
    
    return new_binary_schedule

def swap_mutation_operator(binary_schedule, **kwargs):
    """
    
    *** 24/01/20 LOOK AT THIS IN MORE DETAIL *** 
    
    The swap mutation operation. 
    
    Each hour, randomly choose to either swap bits between two 
    random generators, or change a random generator's status (on/off).
    
    This is applied only to the best genotype each generation.
    """
    T, num_gen = binary_schedule.shape
    new_binary_schedule = np.copy(binary_schedule)
    
    init_status = kwargs.get('init_status')
    
    # initiate best schedule and best fitness
    best_schedule = binary_schedule
    best_fitness = calculate_schedule_fitness(convert_to_integer(binary_schedule, init_status), **kwargs)
    
    for t in range(T):
        new_binary_schedule = np.copy(best_schedule)
        
        # Apply operator
        random = np.random.uniform(0,1)
        if random < 0.5:
            r1, r2 = np.random.choice(np.arange(num_gen), size=2, replace=False)
            temp = np.copy(new_binary_schedule[t, r1])
            new_binary_schedule[t, r1] = new_binary_schedule[t, r2]
            new_binary_schedule[t, r2] = temp
        else:
            r = np.random.choice(np.arange(num_gen))
            new_binary_schedule[t, r] = not new_binary_schedule[t, r]
            
        # Calculate fitness of new schedule
        new_fitness = calculate_schedule_fitness(convert_to_integer(new_binary_schedule, init_status), **kwargs)
        
        # Reassign best fitness and schedule 
        if new_fitness < best_fitness:
            best_schedule = new_binary_schedule
            best_fitness = new_fitness
            
    return best_schedule

def swap_window_hc_operator(binary_schedule, **kwargs):
    """
    Apply the swap window hill-climb operator to a binary schedule. 
    
    Choose random units u1 and u2, and a time window size W. Start at t=0
    and swap the schedules of u1 and u2 in the window [t, t+W]. Retain the new
    schedule only if its fitness improves on the unaltered one. Then increment
    t and repeat with the same u1 and u2 and W. 
    
    Args:
        - binary_schedule (array): schedule to which the operator will be applied. 
        
    Returns:
        - best_schedule (array): the best binary schedule found in applying the operator.
    """
    init_status = kwargs.get('init_status')
    T, num_gen = binary_schedule.shape
    
    # Choose random window size
    W = np.random.choice(np.arange(1,T-1))
    
    # Choose random units
    u1, u2 = np.random.choice(num_gen, size=2, replace=False)
    
    # Assign best schedule and fitness
    best_schedule = binary_schedule
    best_fitness = calculate_schedule_fitness(convert_to_integer(binary_schedule, init_status), **kwargs)
    
    for t in range(T-W):
        # Copy schedule
        new_binary_schedule = np.copy(best_schedule)
        
        # Get window array
        window = np.array(range(t, t+W))
        
        # Swap the schedules for u1 and u2 in the window 
        temp = np.copy(new_binary_schedule[window, u1])
        new_binary_schedule[window, u1] = new_binary_schedule[window, u2]
        new_binary_schedule[window, u2] = temp
        
        # Calculate fitness of new schedule
        new_fitness = calculate_schedule_fitness(convert_to_integer(new_binary_schedule, init_status), **kwargs)
        
        # Reassign best schedule and fitness
        if new_fitness < best_fitness:
            best_schedule = new_binary_schedule
            best_fitness = new_fitness
            
    return best_schedule