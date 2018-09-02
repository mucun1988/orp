import pandas as pd
import numpy as np
import time
import os, sys

sys.path.extend([os.path.join('.', 'sdo')])
sys.path.extend([os.path.join('.', 'mbp')])

from mbp import OneBagPacker
from sdo import OneShelfDisplayOptimizer

try:
    dat_file = '/Users/matthew.mu/dat/StoreDataSetV4.txt'
    dat = pd.read_csv(dat_file, sep='\t')
except:
    dat_file = '/datadrive/matthew/dat/StoreDataSetV4.txt'
    dat = pd.read_csv(dat_file, sep='\t')

L = 45*12/15 #foot
nl = 5

dat['ITEM_WIDTH_QTY'] = dat['ITEM_WIDTH_QTY'].apply(lambda x: round(x,3))
dat['TOT_FACE_QTY_UB'] = dat['ITEM_WIDTH_QTY'].apply(lambda x: int(np.floor(L/x))*nl)
dat['TOT_FACE_QTY_T'] = dat[['TOT_FACE_QTY', 'TOT_FACE_QTY_UB']].min(axis=1)

dat = dat[(dat.TOT_FACE_QTY_T > 2) & (dat.ITEM_WIDTH_QTY > 0)]

skus = dat.OLD_NBR.tolist()
q = dat.TOT_FACE_QTY_T.tolist()
l = dat.ITEM_WIDTH_QTY.tolist()
assert len(q)==len(l)==len(skus)

idx_left = range(len(q))

shelf = 0

for zz in range(2):

    st = time.time()
    # select s to be displayed
    w = [q[i]*l[i] for i in idx_left]
    obp = OneBagPacker(weights=w, capacity=L*nl, dg=1000)
    obp.pack()
    s = [idx_left[i] for i in obp.packed_items] #real index

    # display using products in s

    q_s = [q[i] for i in s]
    l_s = [l[i] for i in s]
    # q_s = [6, 2, 3, 8]
    # l_s = [9.5, 9.5, 9.562, 9.4]
    osdo = OneShelfDisplayOptimizer(q_s, l_s, nl, L, time_limit=-1)
    osdo.optimize()

    idx_put = [s[i] for i in range(len(s)) if osdo.B1d[i] > 0]
    print("**********************************************")
    print(f"summary shelf {shelf}:")
    print(f"  skus to put: {[skus[i] for i in idx_put]}")
    print(f"  time consumed to solve: {time.time()-st}")
    print("**********************************************")
    if osdo.optimal:
        for i in range(len(s)):
            if osdo.B1d[i] > 0:
                print(f"sku {skus[s[i]]}")
            for k in range(nl):
                if osdo.B2d[i][k] > 0:
                    print(f"  layer {k}: position={round(osdo.x[i][k],2)}-{round(osdo.y[i][k],2)}; quantity={osdo.n2d[i][k]}")
    else:
        print('failed!')

    del osdo
    idx_left = [x for x in idx_left if x not in idx_put]
    shelf += 1

