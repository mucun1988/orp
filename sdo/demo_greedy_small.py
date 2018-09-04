import pandas as pd
import numpy as np
import os, sys

sys.path.extend([os.path.join('.', 'sdo')])

from sdo import ShelfDisplayOptimizer


L = 10
nl = 3
m = 10

# skus_info = {'skuA': {'q': 3, 'l': 8},
#              'skuB': {'q': 2, 'l': 1},
#              'skuC': {'q': 2, 'l': 2},
#              'skuD': {'q': 2, 'l': 10},
#              'skuE': {'q': 2, 'l': 5}}
#
# skus_info.update({f'skuF{i}': {'q': 1, 'l': 2} for i in range(15)})

skus_info = {'skuA':{'q':3, 'l':8},
             'skuB1':{'q':1, 'l':2},
             'skuB2':{'q':1, 'l':2},
             'skuB3':{'q':1, 'l':1},
             'skuB4':{'q':1, 'l':1},
             'skuC':{'q':1, 'l':6}}

sdo = ShelfDisplayOptimizer(skus_info,  m, nl, L, time_limit=1000*60*10)
sdo.optimize_greedy()










