#!/usr/bin/env python3
"""
Run the genetic algorithm. 

There are 2 data inputs to the GA:
    
1. demand: should be a .txt file with the demand profile. Can be any length.
2. gen_info: a .csv file with the following variables:
    - min_output: minimum generation (MW)
    - max_output: maximum generation (MW)
    - status: generator's initial status (integer-encoded)
    - a, b, c: coefficients for quadratic fuel cost curve of the form 
    cost = a*E^2 + b*E + c, where E is the energy delivered by the generator in 
    the time period (E = power if periods are 1 hour). 
    - t_min_down: minimum number of periods that generator must spend offline 
    before being turned on
    - t_min_up: minimum number of periods that generator must spend online before 
    being turned off.
    - hot_cost: cost for a hot start ($)
    - cold_cost: cost for a cold start ($)
    - cold_hrs: hot start if downtime (in *periods*) <= cold_hrs, otherwise
    a cold start.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from genetic_algorithm import run_genetic_algorithm
from fitness import calculate_constraint_costs
from economic_dispatch import economic_dispatch


# Change demand profile here
demand = np.genfromtxt('data/ng_scaled_demand_48.txt')

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
              'n_hrs': 0.5, # resolution of settlement periods (1 if hourly, etc.)
              'max_penalty': 1e5}

# Set the number of generations
number_of_generations = 500

# Number of generators and periods
num_gen = gen_info.shape[0]
T = demand.size

# Get a random schedule to begin with
seed_schedules = np.random.choice(2, size = (all_kwargs.get('pop_size'), T, num_gen))

# Run GA
best_genotype, results, population = run_genetic_algorithm(number_of_generations, seed_schedules, **all_kwargs)

# Are constraints violated?
gen_info = all_kwargs.get('gen_info')
init_status = all_kwargs.get('init_status')
penalty = all_kwargs.get('constraint_penalty')
demand = all_kwargs.get('demand')
reserve_margin = all_kwargs.get('reserve_margin')

constraint_costs = calculate_constraint_costs(best_genotype.schedule, gen_info, init_status, penalty, demand, reserve_margin)
print("Constraint costs: {}".format(np.sum(constraint_costs)))

# plot results
fig1, ax1 = plt.subplots()
ax1.plot(np.arange(number_of_generations), results)
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness")
plt.show()

# Calculate ED for the best schedule
ed = economic_dispatch(gen_info, best_genotype.schedule, demand+0.01)

# Plot schedule data frame 
df = pd.DataFrame(ed[0])
df.plot(kind='bar', stacked=True)
plt.show()


