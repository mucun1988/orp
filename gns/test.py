#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:03:53 2018

@author: matthew.mu
"""

import numpy as np
import sys, os

sys.path.extend([os.path.join('.', 'gns')])

from gns import GNS

# hard: # sum_j B_ij = 1 for all i
W = [[1, 10, 100], 
     [1, 20]]
V = [[1,4,30000], 
     [1,2]]

capacity = 2

gns = GNS(V,W,capacity)

gns.solve()

print('Number of variables =', gns.num_variables)
print('Number of constraints =', gns.num_contraints)
print('Optimal objective value = %d' % gns.obj_value)
print('Solution B = ', gns.B)

# hard: # sum_j B_ij = 1 for all i
W = [[1, 10, 100], 
     [1, 20]]
V = [[1,4,30000], 
     [1,2]]

capacity = 101

gns = GNS(V,W,capacity)

gns.solve()

print('Number of variables =', gns.num_variables)
print('Number of constraints =', gns.num_contraints)
print('Optimal objective value = %d' % gns.obj_value)
print('Solution B = ', gns.B)

# soft: # sum_j B_ij <= 1 for all i
W = [[1, 10, 100], 
     [1, 20]]
V = [[1,4,30000], 
     [-1,-2]]
capacity = 2

gns = GNS(V,W,capacity, 'soft')

gns.solve()

print('Number of variables =', gns.num_variables)
print('Number of constraints =', gns.num_contraints)
print('Optimal objective value = %d' % gns.obj_value)
print('Solution B = ', gns.B)

# larger scale
W = np.random.rand(300,10).tolist()
V = np.random.rand(300,10).tolist()
capacity = 100

gns = GNS(V,W,capacity, 'soft')

gns.solve()

print('Number of variables =', gns.num_variables)
print('Number of constraints =', gns.num_contraints)
print('Optimal objective value = %d' % gns.obj_value)

# yg example 
W = [[11, 22, 33], 
     [10, 23]]
V = [[24, 40, 79], 
     [23, 48]]
capacity = 37

gns = GNS(V,W,capacity, 'hard')

gns.solve()

print('Number of variables =', gns.num_variables)
print('Number of constraints =', gns.num_contraints)
print('Optimal objective value = %d' % gns.obj_value)
print('Solution B = ', gns.B)