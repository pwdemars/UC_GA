#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:35:21 2020

@author: patrickdemars
"""

import numpy as np 

from fitness import calculate_start_costs, calculate_fuel_costs
from economic_dispatch import economic_dispatch

def calculate_expected_cost(schedule, gen_info, demand, uncertainty, voll, n_hrs):
    """
    Calculate expected cost of a schedule assuming demand is normally distributed
    around a point forecast.
    
    The expected operating cost is calculated as:
        
        cost = start_cost + sum[fuel_costs(d) + lost_load_costs(d) * probability(d) for d in demand_realisations]
        
    In effect, it is an average of the operating costs under different
    demand realisations, weighted by their probability of occurrence. 
    
    Args:
        - schedule (array): integer-encoded schedule to evaluate 
        - gen_info (data frame): generator specifications: cost curves etc.
        - demand (array): demand profile
        - uncertainty (float): standard deviation is assumed to be uncertainty*demand.
        uncertainty therefore controls levels of stochasticity. 
        - voll (float): value of lost load ($/MWh)
        - n_hrs (float): resolution of demand profile. if half-hourly settlement 
        periods, then set n_hrs=0.5.
    
    Returns:
        - average_cost (float): weighted average of operating costs for the 5 demand
        profiles, weighted by their probability of occurrence.
    """
    T, num_gen = schedule.shape
    
    init_status = gen_info['status'].to_numpy()
    
    
    # Calculate start costs
    start_costs = np.sum(calculate_start_costs(gen_info, schedule, init_status))
    
    # Weights for sum (based on normal distribution CDF)
    sigmas = [0.023, 0.136, 0.682, 0.136, 0.023]

    # Create the demand profiles
    demand_real = np.tile(demand, (5, 1))
    for i, j in enumerate(np.arange(-2, 3)):
        demand_real[i] = demand_real[i] + demand_real[i]*uncertainty*j

    average_cost = start_costs
    
    for sigma, d in zip(sigmas, demand_real):
        # Get economic dispatch and ENS
        dispatch, ens = economic_dispatch(gen_info, schedule, d)
        
        # Get fuel costs
        fuel_costs = np.sum(calculate_fuel_costs(dispatch, gen_info, n_hrs))
        
        # Lost load costs
        lost_load_costs = ens*voll
        
        average_cost += sigma*(fuel_costs + lost_load_costs)
        
    return average_cost
