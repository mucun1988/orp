# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23, 2018
@author: matthew.mu
"""

from sdo import MinBagPacker
from ortools.linear_solver import pywraplp


weights = [8] + [6]*2
capacity = 10
packer = MinBagPacker(weights, capacity, dg=1)
packer.greedy_pack()
packer.bag_weights
packer.packing_result
packer.item_bag_result


def _new_solver(name,integer=False):
  return pywraplp.Solver(name,
    pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
      if integer else pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

def _sol_val(x):
  if type(x) is not list:
    return 0 if x is None \
             else x if isinstance(x,(int,float)) \
                    else x.SolutionValue() if x.Integer() is False \
                                           else int(x.SolutionValue())
  elif type(x) is list:
    return [_sol_val(e) for e in x]

q = [4, 2]
l = [2, 6]
assert len(q)==len(l)
n = len(q)
L = 10.0
nl = 3
m = 2


solver = _new_solver('sdo', integer=True)

B1d = [solver.IntVar(0, 1, f'B_{i}') for i in range(m)]
B2d = [[solver.IntVar(0, 1, f'B_{i}_{j}') for j in range(m)] for i in range(n)]
B3d = [[[solver.IntVar(0, 1, f'B_{i}_{j}_{k}') for k in range(nl)] for j in range(m)] for i in range(n)]

n3d = [[[solver.IntVar(0, solver.infinity(), f'n_{i}_{j}_{k}') for k in range(nl)] for j in range(m)] for i in range(n)]

Left = [[[[solver.IntVar(0, 1, f'L_{i}_{ip}_{j}_{k}') for k in range(nl)] for j in range(m)] for ip in range(n)] for i in range(n)]

x = [[[solver.NumVar(0.0, L, f'x_{i}_{j}_{k}') for k in range(nl)] for j in range(m)] for i in range(n)]
y = [[[solver.NumVar(0.0, L, f'y_{i}_{j}_{k}') for k in range(nl)] for j in range(m)] for i in range(n)]
h = [[[solver.NumVar(0.0, L, f'h_{i}_{j}_{k}') for k in range(nl)] for j in range(m)] for i in range(n)]
t = [[[solver.NumVar(0.0, L, f't_{i}_{j}_{k}') for k in range(nl)] for j in range(m)] for i in range(n)]
o = [[[solver.NumVar(0.0, L, f'o_{i}_{j}_{k}') for k in range(nl)] for j in range(m)] for i in range(n)]

for i in range(n):
  solver.Add(sum([n3d[i][j][k] for j in range(m) for k in range(nl)]) == q[i])

for i in range(n):
  for ip in range(n):
    for j in range(m):
      for k in range(nl):
        if i != ip:
          solver.Add(Left[i][ip][j][k] + Left[ip][i][j][k] + (1-B3d[i][j][k]) + (1 - B3d[ip][j][k]) >= 1)

for i in range(n):
  for ip in range(n):
    for j in range(m):
      for k in range(nl):
        solver.Add(y[i][j][k] + Left[i][ip][j][k] * L <= x[ip][j][k] + L)

for i in range(n):
  solver.Add(sum([B2d[i][j] for j in range(m)])==1)

for i in range(n):
  for j in range(m):
    for k1 in range(nl):
      for k2 in range(nl):
        for k3 in range(nl):
          if k1 < k2 and k2 < k3:
            solver.Add(B3d[i][j][k1] - B3d[i][j][k2] + B3d[i][j][k3] <= 1)

for i in range(n):
  for j in range(m):
    for k in range(nl):
      for kp in range(nl):
        solver.Add(t[i][j][k] - (1-B3d[i][j][k])*L <= y[i][j][kp] + (1-B3d[i][j][kp])*L)

for i in range(n):
  for j in range(m):
    for k in range(nl):
      for kp in range(nl):
        solver.Add(h[i][j][k] + (1-B3d[i][j][k])*L >= x[i][j][kp] - (1-B3d[i][j][kp])*L)

for i in range(n):
  for j in range(m):
    for k in range(nl):
      solver.Add(o[i][j][k] == t[i][j][k] - h[i][j][k])
      solver.Add(o[i][j][k] >= l[i]*B3d[i][j][k])
      solver.Add(y[i][j][k] <= L*B3d[i][j][k])
      solver.Add(t[i][j][k] <= L*B3d[i][j][k])
      solver.Add(h[i][j][k] <= t[i][j][k])
      solver.Add(B3d[i][j][k] <= B2d[i][j])
      solver.Add(y[i][j][k]-x[i][j][k] == l[i]*n3d[i][j][k])

for i in range(n):
  for j in range(m):
    solver.Add(B2d[i][j] <= B1d[j])

solver.Maximize(-sum([B1d[j] for j in range(m)]) + 0.001*sum([o[i][j][k] for i in range(n) for j in range(m) for k in range(nl)]))

solver.Solve()

# output something

x = _sol_val(x)
y = _sol_val(y)
n3d = _sol_val(n3d)

for i in range(n):
    print(f"sku {i}: ")
    for j in range(m):
        print(f"  shelf {j}: ")
        for k in range(nl):
            print(f"    layer {k}: position={round(x[i][j][k],2)}-{round(y[i][j][k],2)}; quantity={n3d[i][j][k]}")


del solver