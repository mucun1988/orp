import pandas as pd
import numpy as np
import os, sys

sys.path.extend([os.path.join('.', 'sdo')])

from sdo import ShelfDisplayOptimizer

try:
    dat_file = '/Users/matthew.mu/dat/StoreDataSetV4.txt'
    dat = pd.read_csv(dat_file, sep='\t')
except:
    dat_file = '/datadrive/matthew/dat/StoreDataSetV4.txt'
    dat = pd.read_csv(dat_file, sep='\t')

L = 45*12/15 #foot
nl = 5
m = 200

dat['ITEM_WIDTH_QTY'] = dat['ITEM_WIDTH_QTY'].apply(lambda x: round(x,3))
dat['TOT_FACE_QTY_UB'] = dat['ITEM_WIDTH_QTY'].apply(lambda x: int(np.floor(L/x))*nl)
dat['TOT_FACE_QTY_T'] = dat[['TOT_FACE_QTY', 'TOT_FACE_QTY_UB']].min(axis=1)

dat = dat[(dat.TOT_FACE_QTY_T > 0) & (dat.ITEM_WIDTH_QTY > 0)]

# input
skus = dat.OLD_NBR.tolist()
q = dat.TOT_FACE_QTY_T.tolist()
l = dat.ITEM_WIDTH_QTY.tolist()
assert len(q)==len(l)==len(skus)

skus_info = dict({str(skus[i]):{'q': q[i],'l': l[i]} for i in range(len(skus))})

sdo = ShelfDisplayOptimizer(skus_info,  m, nl, L, time_limit=1000*60*10)
sdo.optimize_greedy()









