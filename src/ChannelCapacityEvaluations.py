"""
EEE551: Information Theory | Channel Capacity Evaluations

Author: Bharath Keshavamurthy <bkeshav1@asu.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
Copyright (c) 2022. All Rights Reserved.

Reference: https://www.cvxpy.org/examples/applications/Channel_capacity_BV4.57.html
"""

import numpy as np
import cvxpy as cp
from scipy.special import xlogy

"""
CONFIGURATION: Update this according to your eval matrix
"""

# Homework-3 Q2(a)
# transitions = np.array([[0.5, 0.0, 0.5],
#                         [0.5, 0.5, 0.0],
#                         [0.0, 0.5, 0.5]])

# Homework-3 Q2(b)
# transitions = np.vstack([np.arange(start=0.1, stop=1.0, step=0.1),
#                          np.arange(start=0.9, stop=0.0, step=-0.1)])

# Homework-3 Q2(c)
transitions = np.array([[1.0, 0.0, 1 / 3],
                        [0.0, 1.0, 1 / 3],
                        [0.0, 0.0, 1 / 3]])

px_var = cp.Variable(shape=transitions.shape[1])
mutual_info = cp.sum(cp.entr(transitions @ px_var) / np.log(2)) + \
              np.sum(xlogy(transitions, transitions) / np.log(2), axis=0) @ px_var

obj = cp.Maximize(mutual_info)
constraints = [cp.sum(px_var, axis=0) == 1.0, px_var >= 0.0]

problem = cp.Problem(obj, constraints)
problem.solve('SCS', max_iters=int(1e6), eps_abs=1e-6, eps_rel=1e-6, verbose=True)

print(f'[INFO] ChannelCapacityEvaluations main: Problem Status = {problem.status} | '
      f'Optimal Value = {problem.value} | Optimal Input Distribution = {px_var.value}')
