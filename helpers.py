#!/usr/bin/env python3
"""
The helpers module has functions for converting between integer and binary 
encodings of commitment schedules.

The integer encoding is required for determining startup costs and constraint 
violations, as it includes the on/off times for each generator at each hour.
"""

import numpy as np

def convert_to_binary(integer_schedule):
    """
    Converts an integer-encoded schedule to a binary encoded one. 
    
    Args:
        - integer_schedule (array): a T by N array where each cell gives the
        on (positive) or off (negative) time for that generator.
        
    Returns:
        - binary_schedule (array): a T by N array where each cell indicates whether
        the generator is online (1) or offline (0).
    """
    binary_schedule = np.where(integer_schedule > 0, 1, 0)
    return binary_schedule

def convert_to_integer(binary_schedule, init_status):
    """
    Converts a binary-encoded schedule to an integer-encoded one. 
    
    This is an inelegant double for-loop over time periods and generators. 
    Essentially, at each time step we know the binary values for t-1 and t
    which we will denote as b0 and b1 respectively, as well as the integer value
    for t-1 which we denote as i0. The task is to find i1s for each generator 
    at each iteration.
    
    Args:
        - binary_schedule (array): a T by N array where each cell indicates whether
        the generator is online (1) or offline (0).
    
    Returns:
        - integer_schedule (array): a T by N array where each cell gives the
        on (positive) or off (negative) time for that generator.
    """
    T = binary_schedule.shape[0]
    num_gen = binary_schedule.shape[1]
    binary_init_status = np.where(init_status > 0, 1, 0) # binarised init_status
    integer_schedule = np.zeros((T, num_gen))
    
    for t in range(T):
        for n in range(num_gen):
            if t == 0: 
                i0 = init_status[n]
                b0 = binary_init_status[n]
            else: 
                i0 = integer_schedule[t-1,n]
                b0 = binary_schedule[t-1,n]
            b1 = binary_schedule[t,n]
                        
            # If both binary are the same (not turned on or off), then just add 1 or -1.  
            if b0==b1: # On in consecutive periods, or off in consecutive periods
                integer_schedule[t,n] = i0 + (1 if b1 > 0 else -1)
            # If turning off, set value to -1
            elif b0 > b1:
                integer_schedule[t,n] = -1
            else:
                integer_schedule[t,n] = 1
    return integer_schedule