import numpy as np
from pyDOE import lhs
def latin_hypercube_sampling_ichains(min_vals, max_vals, num_chains):
    sampler = lhs(len(min_vals),samples=num_chains)
    initial_params = np.zeros_like(sampler)
    for i, (low, high) in enumerate(zip(min_vals, max_vals)):
        initial_params[:, i] = low + sampler[:, i] * (high - low)
    return initial_params