import os, sys, time
sys.path.extend([os.path.join('.', 'sdo')])
from sdo import ShelfDisplayOptimizer

st = time.time()

# input
skus_info = {'skuA': {'q': 10, 'l':2},
             'skuB': {'q': 5, 'l': 5},
             'skuC': {'q': 4, 'l': 1},
             'skuD': {'q': 10, 'l': 2}}

m = 2   # number of shelves available
nl = 5     # number of layers of each shelf
L = 10    # length of each layer

sdo = ShelfDisplayOptimizer(skus_info, m, nl, L)

sdo.optimize()

if sdo.optimal:
    print("**********************************************")
    print("summary:")
    print(f"  shelves to use: {[i for i in range(m) if sdo.B1d[i]>0]}")
    print(f"  time consumed to solve: {time.time()-st}")
    print("**********************************************")
    for id in skus_info.keys():
        print(f"sku {id}: ")
        result = sdo.result[id]
        for i in range(len(result['n'])):
            print(f" shelf {result['shelf'][i]}-layer {result['layer'][i]}:")
            print(f"     position={round(result['x'][i], 2)}-{round(result['y'][i], 2)}; quantity={result['n'][i]}")
else:
    print('failed!')

