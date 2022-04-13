"""
EEE551 HW-5: Cost-Constrained Binary Channels | Q1

Author: Bharath Keshavamurthy <bkeshav1@asu.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
Copyright (c) 2022. All Rights Reserved.

Reference: https://www.cvxpy.org/examples/applications/Channel_capacity_BV4.57.html
"""

# The imports
import numpy as np
import cvxpy as cp
from scipy.special import xlogy

"""
CONFIGURATIONS
"""
cp.set_num_threads(8)
p = 0.1  # The error or erasure probability
cost = 1.0  # The cost associated with the mean of the binary input distribution

"""
UTILITIES
"""
entropy_fn = lambda _p: -(_p * np.log2(_p)) - ((1 - _p) * np.log2(1 - _p))

"""
THEORETICAL EVALUATION (From HW-5 'Optimization Theory' Constructions)
"""

'''
Binary Symmetric Channel
'''

# channel_type = 'BSC'
# p_ = 1.0 - p + cost * (2.0 * p - 1.0)
# transitions = np.array([[1 - p, p], [p, 1 - p]])
# px = (lambda: [0.5, 0.5], lambda: [1 - cost, cost])[0.0 <= cost <= 0.5]()
# theoretical_capacity = (lambda: 1.0 - entropy_fn(p), lambda: entropy_fn(p_) - entropy_fn(p))[0.0 <= cost <= 0.5]()

'''
Binary Erasure Channel
'''

channel_type = 'BEC'
transitions = np.array([[1 - p, 0.0], [p, p], [0.0, 1 - p]])
px = (lambda: [0.5, 0.5], lambda: [1 - cost, cost])[0.0 <= cost <= 0.5]()
theoretical_capacity = (lambda: 1 - p, lambda: (1 - p) * entropy_fn(cost))[0.0 <= cost <= 0.5]()

"""
EMPIRICAL EVALUATION
"""

x = cp.Parameter(value=np.array([0, 1]), shape=(2,))
px_var = cp.Variable(value=np.zeros(shape=(2,)), shape=(2,))
mutual_info = cp.sum(cp.entr(transitions @ px_var) / np.log(2)) + \
              np.sum(xlogy(transitions, transitions) / np.log(2), axis=0) @ px_var

# noinspection PyTypeChecker
problem = cp.Problem(objective=cp.Maximize(mutual_info),
                     constraints=[cp.sum(px_var, axis=0) == 1.0,
                                  px_var >= 0.0, px_var @ x <= cost])
problem.solve('SCS', max_iters=int(1e8), eps_abs=1e-8, eps_rel=1e-8, verbose=True)
empirical_capacity = problem.value

"""
RESULTS
"""

print(f'[INFO] CostConstrainedBinaryChannels | {channel_type}')

print(f'[INFO] CostConstrainedBinaryChannels CapacityEvaluations empirical: '
      f'Status = {problem.status} | Channel Capacity = {np.round(problem.value, 6)} bits | '
      f'Achieving Distribution = {np.round(px_var.value, 6)} | Given Cost = {cost} | '
      f'Cost = {np.round((px_var @ x).value, 6)}.')

print('[INFO] CostConstrainedBinaryChannels CapacityEvaluations theoretical: '
      f'Channel Capacity = {np.round(theoretical_capacity, 6)} bits | Achieving Distribution = {np.round(px, 6)}.')
