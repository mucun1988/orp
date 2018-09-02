import os, sys, time
sys.path.extend([os.path.join('.', 'sdo')])
from sdo import ShelfDisplayOptimizer

st = time.time()

q = [10, 5, 4, 10]  # product quantities
l = [2, 5, 1, 2]  # length of each product
n = len(q)  # number of products to be displayed on shelf
m = 2   # number of shelves available
nl = 5     # number of layers of each shelf
L = 10    # length of each layer

sdo = ShelfDisplayOptimizer(q,l,m,nl,L)

sdo.optimize()

if sdo.optimal:
    print("**********************************************")
    print("summary:")
    print(f"  shelves to use: {[i for i in range(m) if sdo.B1d[i]>0]}")
    print(f"  time consumed to solve: {time.time()-st}")
    print("**********************************************")
    for i in range(n):
        print(f"sku {i}: ")
        for j in range(m):
            if sdo.B2d[i][j] > 0:
                print(f"  shelf {j}: ")
            for k in range(nl):
                if sdo.B3d[i][j][k] > 0:
                    print(f"    layer {k}: position={round(sdo.x[i][j][k],2)}-{round(sdo.y[i][j][k],2)}; quantity={sdo.n3d[i][j][k]}")
else:
    print('failed!')