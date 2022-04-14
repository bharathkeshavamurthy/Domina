"""
EEE551 HW-5: Cost-Constrained BEC | Q3

Author: Bharath Keshavamurthy <bkeshav1@asu.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
Copyright (c) 2022. All Rights Reserved.
"""

# The imports
import numpy as np
import cvxpy as cp

"""
CONFIGURATIONS
"""
cp.set_num_threads(8)
cost, number_of_channels, pe_s = 1.0, 3, np.array([0.1, 0.4, 0.7])

"""
UTILITIES
"""
entropy_fn = lambda _p: (-_p * np.log2(_p)) + (-(1 - _p) * np.log2(1 - _p))

"""
THEORETICAL EVALUATION
"""

inv_pe_s = 1 / (1 - pe_s)


def ps_fn(alpha):
    ps = np.array([1 / (1 + (alpha ** _i_pe)) for _i_pe in inv_pe_s])
    ps[ps < 0] = 0
    ps[ps > 1] = 1
    return ps


def bisect(low, high, tol, max_confidence):
    mid, confidence, converged = 0.0, 0, False
    cost_th = cost if cost <= 0.5 * number_of_channels else 0.5 * number_of_channels
    while not converged or confidence < max_confidence:
        mid = (low + high) / 2
        ps = ps_fn(mid)
        if np.sum(ps) < cost_th:
            high = mid
        else:
            low = mid
        converged = abs(high - low) < tol
        confidence += 1 if converged else -confidence
    ps = ps_fn(mid)
    return np.sum([(1 - pe_s[i]) * entropy_fn(ps[i]) for i in range(number_of_channels)]), ps


cap, pxs = bisect(1.0, 1e4, 1e-8, 10)
theoretical_capacity, theoretical_pxs = np.round(cap, 6), np.round(pxs, 6)

"""
EMPIRICAL EVALUATION
"""

px_vars = cp.Variable(shape=(number_of_channels,))

# noinspection PyTypeChecker
problem = cp.Problem(objective=cp.Maximize((1 - pe_s) @ ((cp.entr(px_vars) / np.log(2)) +
                                                         (cp.entr(1.0 - px_vars) / np.log(2)))),
                     constraints=[px_vars >= 0.0, px_vars <= 1.0, cp.sum(px_vars) - cost <= 0.0])

problem.solve('SCS', max_iters=int(1e8), eps_abs=1e-8, eps_rel=1e-8, verbose=True)
status, empirical_capacity, empirical_pxs = problem.status, np.round(problem.value, 6), np.round(px_vars.value, 6)

"""
RESULT
"""

print(f'[INFO] CostConstrainedParallelBEC empirical: Channel Capacity Maximization for parallel BECs | '
      f'Problem Status = {status} | Capacity = {empirical_capacity} bits | '
      f'Achieving Bernoulli Parameters = {empirical_pxs} | Given cost = {cost} | '
      f'Cost = {np.round(cp.sum(px_vars).value, 6)}.')

print(f'[INFO] CostConstrainedParallelBEC theoretical: Channel Capacity = {theoretical_capacity} bits | '
      f'Achieving Bernoulli Parameters = {theoretical_pxs}.')
