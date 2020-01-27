#!/usr/bin/env python3
"""
Run the genetic algorithm.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(1, '/Users/patrickdemars/Documents/Projects/UC_GA/UC_GA')

from genetic_algorithm import run_genetic_algorithm
from helpers import convert_to_integer


# Change demand profile here
demand = np.genfromtxt('data/kazarlis_demand.txt')

# Change gen_info here
gen_info = pd.read_csv('data/kazarlis_units.csv')

# Supply the kwargs
all_kwargs = {'demand': demand,
              'gen_info': gen_info,
              'init_status': gen_info['status'],
              'voll': 1e3,
              'constraint_penalty': 1e4,
              'reserve_margin': 0.1,
              'mutation_probability': 0.01,
              'crossover_probability': 0.5,
              'swap_window_probability': 0.3,
              'window_mutation_probability': 0.3,
              'swap_window_hc_probability': 0.3,
              'pop_size': 50,
              'max_penalty': 1e4}

# Set the number of generations
number_of_generations = 20

# Number of generators and periods
num_gen = gen_info.shape[0]
T = demand.size

# Get a random schedule to begin with
random_schedule = np.random.choice(2, size = (T, num_gen))
init_status = all_kwargs.get('init_status')
seed_schedule = convert_to_integer(random_schedule, init_status)

# Run GA
best_genotype, results_3 = run_genetic_algorithm(number_of_generations, seed_schedule, **all_kwargs)