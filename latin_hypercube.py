import numpy as np
from pyDOE import lhs

def latin_hypercube_sampling_ichains(min_vals, max_vals, num_chains):
    #Generate a sample for values between 0 and 1, dividing the space into num_chains intervals of the same size
    #Take exactly one point for each interval, to achieve a well-distributed sample
    sampler = lhs(len(min_vals),samples=num_chains)
    initial_params = np.zeros_like(sampler)
    #It scales from the [0,1] interval to the real one, gived by the minimum and maximum values.
    for i, (low, high) in enumerate(zip(min_vals, max_vals)):
        initial_params[:, i] = low + sampler[:, i] * (high - low)
    return initial_params