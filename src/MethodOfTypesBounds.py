"""
EEE551 HW-6 Q1

Evaluating the lower & upper bounds for the type class size on a given alphabet for sequences of a certain length.

Author: Bharath Keshavamurthy <bkeshav1@asu.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
Copyright (c) 2022. All Rights Reserved.
"""

# The imports
import math
import numpy as np
import pandas as pd
from functools import reduce
from tabulate import tabulate
from itertools import product
from scipy.special import xlogy

"""
CONFIGURATIONS: The alphabet size | The sequence length

TODO: With the alphabet_size as the only alphabet-related configuration, the alphabet is by default set to 
{0, 1, 2, ..., alphabet_size} (typical), so add a remapping block towards the end of the script for atypical alphabets.
"""
alphabet_size, sequence_length = 2, 20
assert alphabet_size > 0 and sequence_length > 0

"""
UTILITIES
"""

size_e = lambda _p: math.comb(sequence_length, int(_p[0] * sequence_length))
entropy = lambda _p: reduce(lambda __p, __q: __p + __q, -xlogy(_p, _p) / np.log(2))
size_t = lambda _p: math.factorial(sequence_length) / reduce(lambda __p, __q: __p * __q,
                                                             [math.factorial(sequence_length * __p) for __p in _p])

upper_bound = lambda _p: 2 ** (sequence_length * entropy(_p))
lower_bound = lambda _p: (2 ** (sequence_length * entropy(_p))) / ((sequence_length + 1) ** alphabet_size)

"""
Construction of the set of types
"""

alphabet = range(alphabet_size)
""" FIXME: Come up with a more efficient way to do this construction. """
itr = np.unique([[''.join([str(_e) for _e in _tup]).count(str(_x)) / sequence_length
                  for _x in alphabet] for _tup in product(range(alphabet_size), repeat=sequence_length)], axis=0)

"""
Evaluation of type class size and its lower & upper bounds
"""
dataframe = pd.DataFrame([[_p, lower_bound(_p), size_t(_p), size_e(_p), size_t(_p) == size_e(_p),
                           upper_bound(_p), lower_bound(_p) <= size_e(_p) <= upper_bound(_p)] for _p in itr],
                         columns=['Type (P)', 'Lower Bound', 'Theoretical |T(P)|', 'Empirical |T(P)| nCk (comb)',
                                  'Theoretical = Empirical ?', 'Upper Bound', 'Lower Bound <= |T(P)| <= Upper Bound ?'])

"""
Pretty print the pandas dataframe
"""
print('[INFO] MethodOfTypesBounds main: Evaluating the bounds for the type class size |T(P)| - \n{}'.format(
    tabulate(dataframe, headers='keys', tablefmt='psql', showindex=False, numalign='center', stralign='center')))
