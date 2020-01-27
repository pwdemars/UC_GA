#!/usr/bin/env python3
"""
The genotype module includes the Population() and Genotype() classes, which are 
the building blocks of the genetic algorithm. 

Genotype objects hold a schedule and fitness as attributes. 

Population objects hold several Genotypes and a new one is created in generation
of a genetic algorithm.
"""
import numpy as np

from fitness import *

class Population(object):
    """
    A collection of genotypes.
    """
    def __init__(self, size):
        self.size = size
        self.num_used = 0
        self.genotypes = []
    
    def add_genotype(self, genotype):
        """
        Add a genotype to the list of genotypes. Increment num_used.
        """
        self.genotypes.append(genotype)
        self.num_used += 1
    
    def remove_genotype(self, genotype):
        """
        Remove a genotype from the list of genotypes. Decrement num_used.
        """
        self.genotypes.remove(genotype)
        self.num_used -= 1
        
    def best_two_genotypes(self): 
        """
        Get the indexes of the best two genotypes (genotypes with best fitness (lowest cost)). 
         
        Returns a list with the two best genotypes. 
        """
        fitnesses = np.array([x.fitness for x in self.genotypes])
        best_idxs = np.argsort(fitnesses)[:2]
        return list(np.array(self.genotypes)[best_idxs])
    
    def reset(self):
        self.genotypes = []
        self.num_used = 0

class Genotype(object):
    """
    The genotype object holds a single (integer) schedule and its fitness. Several genotypes
    make up a population.
    """
    def __init__(self, schedule, **kwargs):
        self.schedule = schedule
        self.demand = kwargs.get('demand')
        self.gen_info = kwargs.get('gen_info')
        self.init_status = kwargs.get('init_status')
        self.voll = kwargs.get('voll')
        self.constraint_penalty = kwargs.get('constraint_penalty')
        self.reserve_margin = kwargs.get('reserve_margin')
        
        # Calculate the fitness
        self.fitness = self.calculate_fitness()

    
    def calculate_fitness(self):
        """
        Calculate the fitness of a genotype which is the sum of the fuel costs, lost load costs,
        start costs and constraint violation costs for that schedule. 
        
        Returns:
            - fitness (float): fitness 
        """
        # Get economic dispatch and ENS
        dispatch, ens = economic_dispatch(self.gen_info, self.schedule, self.demand)
        # Calculate fuel costs
        fuel_costs = np.sum(calculate_fuel_costs(dispatch, self.gen_info))
        # Calculate lost load costs
        lost_load_costs = np.sum(ens*self.voll)
        # Calculate start costs
        start_costs = np.sum(calculate_start_costs(self.gen_info, self.schedule, self.init_status))
        # Calculate constraint costs 
        constraint_costs = calculate_constraint_costs(self.schedule, self.gen_info, 
                                                      self.init_status, self.constraint_penalty,
                                                      self.demand, self.reserve_margin)
        
        # Sum to get fitness
        fitness = np.sum(fuel_costs) + np.sum(lost_load_costs) + np.sum(start_costs) + np.sum(constraint_costs)
        
        return fitness 