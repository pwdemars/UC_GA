#!/usr/bin/env python3
"""
The fitness module gives all the cost functions required to calculate 
the schedule fitness. In particular, it comprises 4 main functions:

1. calculate_fuel_costs()
2. calculate_start_costs()
3. calculate_constraint_costs()

The lost load costs are calculated from the energy not served (ENS) which
is taken from the economic_dispatch function ().
"""

from economic_dispatch import economic_dispatch

import numpy as np

def calculate_schedule_fitness(schedule, **kwargs):
    """
    Calculate the fitness of a schedule as the sum of its fuel costs,
    start costs, lost load costs and constraint costs.

    Args:
        - schedule (array): integer-encoded schedule where each cell
        gives the times of activity/inactivity for that generator.

    Returns: 
        - fitness (float): the evaluated fitness function corresponding 
        to operating cost and costs associated with constraint violations. 
    """
    gen_info = kwargs.get('gen_info')
    demand = kwargs.get('demand')
    voll = kwargs.get('voll')
    init_status = kwargs.get('init_status')
    constraint_penalty = kwargs.get('constraint_penalty')
    reserve_margin = kwargs.get('reserve_margin')
    n_hrs = kwargs.get('n_hrs')
    
    # Get economic dispatch and ENS
    dispatch, ens = economic_dispatch(gen_info, schedule, demand)
    
    # Calculate fuel costs
    fuel_costs = np.sum(calculate_fuel_costs(dispatch, gen_info, n_hrs))
    
    # Calculate lost load costs
    lost_load_costs = np.sum(ens*voll)
    
    # Calculate start costs
    start_costs = np.sum(calculate_start_costs(gen_info, schedule, init_status))
    
    # Calculate constraint costs 
    constraint_costs = calculate_constraint_costs(schedule, gen_info, init_status, 
                                                  constraint_penalty, demand, reserve_margin)

    # Sum costs to get fitness
    fitness = np.sum(fuel_costs) + np.sum(lost_load_costs) + np.sum(start_costs) + np.sum(constraint_costs)
    
    return fitness

def calculate_fuel_costs(dispatch, gen_info, n_hrs):
    """
    Calculate the fuel costs over all the periods in a dispatch schedule 
    of dimension (T,N). 
    
    Args:
        - dispatch (array): an array of unit dispatches of dimension (T,N)
        - gen_info (data frame): generator specs
        
    Returns:
        - fuel_costs (array): array of length T with period fuel costs. 
    """
    T = dispatch.shape[0]
    fuel_costs = np.zeros(T)
    for t in range(T):
        dispatch_period = dispatch[t]
        costs_period = calculate_costs_period(dispatch_period, gen_info, n_hrs)
        fuel_costs[t] = np.sum(costs_period)
    return fuel_costs
        
def calculate_costs_period(outputs, gen_info, n_hrs):
    """Calculate production costs for units in a  given period. 
    Quadratic cost curves are of the form: cost = (a^2(x) + b(x) + c)*time_in_hours
    
    Args:
      - outputs: array of generating outputs
      - a, b, c: arrays of coefficients for quadratic cost curves
      - n_hours: resolution of settlement periods. 0.5 if half-hourly etc. 
    
    Outputs:
        - cost_list: a list of production costs for each unit. 
    """
    num_gen = gen_info.shape[0]
    a = gen_info['a'].to_numpy()
    b = gen_info['b'].to_numpy()
    c = gen_info['c'].to_numpy()
    costs = np.zeros(num_gen)
    for i in range(num_gen):
        if outputs[i] == 0:
            costs[i] = 0
        else:      
            cost_unit = n_hrs*(a[i]*(outputs[i]**2) + b[i]*outputs[i] + c[i])
            costs[i] = cost_unit
    return costs

def calculate_start_costs(gen_info, schedule, init_status):
    """
    Calculate the start costs of a schedule given an initial status preceding 
    the schedule.
    
    Args:
        - gen_info (data frame): generator specs
        - schedule (array): array of shape (N, T) where N is the number of generators
        and T is the number of time periods. It is a binary description of the on/off
        schedules of the generators
        - init_status (array): array of size N, the unit statuses in the period
        preceding the schedule.     
    """
    T = schedule.shape[0]
    start_costs = np.zeros(T)
    for t in range(T):
        if t == 0:
            start_costs_period = calculate_start_costs_period(gen_info, schedule[t], init_status)
        else:
            start_costs_period = calculate_start_costs_period(gen_info, schedule[t], schedule[t-1])
        start_costs[t] = np.sum(start_costs_period)
    return(start_costs)

def calculate_start_costs_period(gen_info, status, prev_status):
    """
    Calculate the start costs between consecutive periods. Used in calculate_start_costs()
    to calculate start costs for an entire schedule. 
    
    Args:
        - gen_info (data frame): generator specs
        - status (array): status at time N 
        - prev_status (array): status at time N-1
        
    Returns:
        - start_costs (array): array of length N, the start costs for each unit. 
    """
    num_gen = gen_info.shape[0]
    cold_hrs = gen_info['cold_hrs'].to_numpy()
    cold_cost = gen_info['cold_cost'].to_numpy()
    hot_cost = gen_info['hot_cost'].to_numpy()
    
    action = np.where(status > 0, 1, 0)
    prev_action = np.where(prev_status > 0, 1, 0)
    idx = [list(map(list, zip(action, prev_action)))[i] == [1,0] for i in range(num_gen)]
    idx = np.where(idx)[0]
    
    start_costs = np.zeros(num_gen)
    
    for i in idx:
        if abs(prev_status[i]) <= -cold_hrs[i]: #####
            start_costs[i] = hot_cost[i]
        else:
            start_costs[i] = cold_cost[i]
    return start_costs

def calculate_constraint_costs(integer_schedule, gen_info, init_status, penalty, demand, reserve_margin):
    """
    Calculate the costs resulting from constraint violations. This is done by simply 
    counting the number of times a minimum up/down time constraint has been violated,
    and multiplying by the penalty.
    
    Args:
        - gen_info (data frame): generator specs
        - integer_schedule (array): array of shape (N, T) where N is the number of generators
        and T is the number of time periods. It is the integer encoding of the schedule (gives)
        on/off times. 
        - init_status (array): array of size N, the unit statuses in the period
        preceding the schedule.
        
    Returns:
        - constraint_costs (array): array of size T, giving the constraint costs for 
        each period.
    """    
    T = integer_schedule.shape[0]
    constraint_costs = np.zeros(T)
    for t in range(T):
        if t == 0:
            constraint_costs_period = calculate_constraint_costs_period(gen_info, integer_schedule[t], 
                                                                        init_status, penalty, 
                                                                        reserve_margin, demand[t])
        else:
            constraint_costs_period = calculate_constraint_costs_period(gen_info, integer_schedule[t], 
                                                                        integer_schedule[t-1], penalty, 
                                                                        reserve_margin, demand[t])
        constraint_costs[t] = np.sum(constraint_costs_period)
    return(constraint_costs)

def calculate_constraint_costs_period(gen_info, status, prev_status, penalty, reserve_margin, demand_period):
    """
    Calculate the number of constraint violations between consecutive periods.
    cmax(aw)
    Args:
        - gen_info (data frame): generator specs
        - status (array): status at time N 
        - prev_status (array): status at time N-1
        - penalty (float): the penalty ($) for each constraint violation
        - reserve_margin (float): proportion of load to add as reserve (e.g. 0.1)
        - demand_period (float): demand for the given period 
        
    Returns:
        - constraint_costs (float): the constraint costs for that period. 
    """
    num_gen = gen_info.shape[0]
    t_min_down = gen_info['t_min_down'].to_numpy()
    t_min_up = gen_info['t_min_down'].to_numpy()

    # Convert schedule to binary 
    action = np.where(status > 0, 1, 0)
    prev_action = np.where(prev_status > 0, 1, 0)
    
    min_down_count = 0 # count for minimum down time violations
    min_up_count = 0 # count for minimum up time violations
    
    # Minimum down time violations
    on_idx = [list(map(list, zip(action, prev_action)))[i] == [1,0] for i in range(num_gen)]
    on_idx = np.where(on_idx)[0]
    for i in on_idx:
        if abs(prev_status[i]) < t_min_down[i]:
            min_down_count += 1
    
    # Minimum up time violations
    off_idx = [list(map(list, zip(action, prev_action)))[i] == [0,1] for i in range(num_gen)]
    off_idx = np.where(off_idx)[0]
    for i in off_idx:
        if abs(prev_status[i]) < t_min_up[i]:
            min_up_count += 1
            
    # Reserve constraint violations
    if np.dot(action, gen_info['max_output']) < (1 + reserve_margin)*demand_period:
        reserve_violation = 1
    else:
        reserve_violation = 0
    
    total_count = min_down_count + min_up_count + reserve_violation
    constraint_costs = total_count*penalty
    return constraint_costs
