#!/usr/bin/env python3
"""
The economic_dispatch module contains all the functions needed to calculate economic 
dispatch for a schedule. The three required inputs for these modules are the following:

1. a binary schedule of on/off statuses for generators 
2. generator cost curves
3. demand profile
"""

import numpy as np

def economic_dispatch(gen_info, schedule, demand):
    """
    Calculates the economic dispatch for a schedule (binary) and demand profile
    given a set of generator specifications. 
    
    Args:
        - gen_info (data frame): df of generator specs with fuel cost curves
        - schedule (array): array of shape (N, T) where N is the number of generators
        and T is the number of time periods. It is a binary description of the on/off
        schedules of the generators
        - demand profile (array): 1D array of size T with the absolute demands (MWh) for
        each time period.
    
    Returns: 
        - dispatch (array): array of shape (N, T) giving the absolute outputs of the 
        generators. 
        - ens (array): 1D array of size T giving the absolute difference between demand
        and supply.
    """
    T = len(demand)
    N = gen_info.shape[0]
    dispatch = np.zeros(shape=(T,N))
    ens = 0
    for t in range(T):
        schedule_t = schedule[t,:]
        demand_t = demand[t]
        disp_t, ens_t = economic_dispatch_period(gen_info, schedule_t, demand_t)
        dispatch[t,:] = disp_t
        ens += ens_t
    return dispatch, ens

def economic_dispatch_period(gen_info, schedule, demand):
    """
    The economic dispatch for a single period. Used in economic_dispatch() to get the
    ED for a whole schedule. 
    """
    action = np.where(schedule > 0, 1, 0)
    idx = np.where(np.array(action) == 1)[0]
    on_a = np.array(gen_info['a'][idx])
    on_b = np.array(gen_info['b'][idx])
    on_min = np.array(gen_info['min_output'][idx])
    on_max = np.array(gen_info['max_output'][idx]) 
    disp = np.zeros(gen_info.shape[0])
    ens = 0
    lambda_lo = 0
    lambda_hi = 30
    if np.sum(on_max) < demand:
        econ = on_max
        ens = demand - np.sum(on_max)
    elif np.sum(on_min) > demand:
        econ = on_min
        ens = np.sum(on_min) - demand
    else:
        econ = lambda_iteration(demand, lambda_lo,
                                lambda_hi, on_a, on_b,
                                on_min, on_max, epsilon=0.1)
    for (i, e) in zip(idx, econ):
        disp[i] = e
    return disp, ens
    
    
def lambda_iteration(load, lambda_low, lambda_high, a, b, mins, maxs, epsilon):
    """Calculate economic dispatch using lambda iteration. 
    
    Args:
        - load: the demand to be met 
        - lambda_low, lambda_high: initial lower and upper values for lambda
        - a: coefficients for quadratic load curves
        - b: constants for quadratic load curves
        - epsilon: stopping criterion for the iteration.  
    
    Returns: 
    a list of outputs for the generators.
    """
    num_gen = len(a)
    lambda_low = np.float(lambda_low)
    lambda_high = np.float(lambda_high)
    lambda_mid = 0
    total_output = sum(calculate_loads(lambda_high, a, b, mins, maxs, num_gen))
    while abs(total_output - load) > epsilon:
        lambda_mid = (lambda_high + lambda_low)/2
        total_output = sum(calculate_loads(lambda_mid, a, b, mins, maxs, num_gen))
        if total_output - load > 0:
            lambda_high = lambda_mid
        else:
            lambda_low = lambda_mid
    return calculate_loads(lambda_mid, a, b, mins, maxs, num_gen)

def calculate_loads(lm, a, b, mins, maxs, num_gen):
    """Calculate loads for all generators as a function of lambda.
    lm: lambda
    a, b: coefficients for quadratic curves of the form cost = a^2p + bp + c
    num_gen: number of generators
    
    Returns a list of individual generator outputs. 
    """
    powers = []
    for i in range(num_gen):
        p = (lm - b[i])/a[i]
        if p < mins[i]:
            p = mins[i]
        elif p > maxs[i]:
            p = maxs[i]
        powers.append(p)
    return powers
