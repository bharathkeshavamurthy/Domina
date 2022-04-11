"""
EEE551 HW-5: Parallel Gaussian Channels | Q2

Author: Bharath Keshavamurthy <bkeshav1@asu.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
Copyright (c) 2022. All Rights Reserved.

Reference: https://www.cvxpy.org/examples/applications/water_filling_BVex5.2.html
"""

# The imports
import numpy as np
import cvxpy as cp

"""
CONFIGURATIONS
"""
cp.set_num_threads(8)
max_power, number_of_channels, noise_powers = 15.0, 3, np.array([1.0, 4.0, 10.0])

"""
THEORETICAL EVALUATION
"""


def signal_powers(alpha):
    powers = alpha - noise_powers
    powers[powers < 0.0] = 0.0
    return powers


def bisect(low, high, tol, max_confidence):
    ns = noise_powers
    mid, confidence, converged = 0.0, 0, False
    while not converged or confidence < max_confidence:
        mid = (low + high) / 2
        ps = signal_powers(mid)
        if np.sum(ps) < max_power:
            low = mid
        else:
            high = mid
        converged = abs(high - low) < tol
        confidence += 1 if converged else -confidence
    return np.round(np.sum(0.5 * np.log2(1 + (signal_powers(mid) / ns))), 6), np.round(signal_powers(mid), 6)


theoretical_capacity, theoretical_pwrs = bisect(1e-4, 1e4, 1e-8, 10)

"""
EMPIRICAL EVALUATION
"""

inv_n_s = cp.Parameter(value=1 / noise_powers, shape=(number_of_channels,))
p_s = cp.Variable(value=np.zeros(shape=number_of_channels), shape=(number_of_channels,))

# noinspection PyTypeChecker
problem = cp.Problem(constraints=[p_s >= 0.0, cp.sum(p_s) - max_power <= 0.0],
                     objective=cp.Maximize(cp.sum(cp.multiply(0.5 / np.log(2.0),
                                                              cp.log(1.0 + cp.multiply(p_s, inv_n_s))), axis=0)))

problem.solve('SCS', max_iters=int(1e8), eps_abs=1e-8, eps_rel=1e-8, verbose=True)
status, empirical_capacity, empirical_pwrs = problem.status, np.round(problem.value, 6), np.round(p_s.value, 6)

"""
RESULTS
"""

print(f'[INFO] PowerAllocation empirical: Channel Capacity Maximization for parallel Gaussian channels | '
      f'Problem Status = {status} | Capacity = {empirical_capacity} bits | '
      f'Optimal powers = {empirical_pwrs}.')

print(f'[INFO] PowerAllocation theoretical: Channel Capacity Maximization for parallel Gaussian channels | '
      f'Capacity = {theoretical_capacity} bits | Optimal powers = {theoretical_pwrs}.')
