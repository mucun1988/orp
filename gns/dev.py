#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:10:23 2018

@author: matthew.mu
"""

from ortools.linear_solver import pywraplp

W = [[1, 10, 100], [1, 20]]
V = [[1,4,30000], [1,2]]
capacity = 2
 
solver = pywraplp.Solver('SolveIntegerProblem',
                pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)


# B_ij is binary
for i in range(len(W)):
    for j in range(len(W[i])):
        exec("B_" + str(i) + str(j) + " = solver.IntVar(0, 1, 'B_" + str(i) + str(j) + "')")

# sum_j B_ij = 1 for all i
for i in range(len(W)):
  # constraint_i = solver.Constraint(1, 1)
  exec("constraint_" + str(i) + "= solver.Constraint(1, 1)")
  for j in range(len(W[i])):
      # constraint_i.SetCoefficient(B_ij, 1)
      exec("constraint_" + str(i) + ".SetCoefficient(B_" + str(i) + str(j) + ", 1)")
      
# sum_{ij} W_ij B_ij <= capacity
constraint_capacity = solver.Constraint(-solver.infinity(), capacity)
for i in range(len(W)):
    for j in range(len(W[i])):
        # constraint_capacity.SetCoefficient(B_ij, W[i][j])
        exec("constraint_capacity.SetCoefficient(B_" + str(i) + str(j) + ", W[" + str(i) + "][" + str(j) + "])")
        
# obj: max sum_i sum_j V_ij B_ij
objective = solver.Objective()
for i in range(len(W)):
    for j in range(len(W[i])):
        # objective.SetCoefficient(B_ij, V[i][j])
        exec("objective.SetCoefficient(B_" + str(i) + str(j) + ", V[" + str(i) + "][" + str(j) + "])")
objective.SetMaximization()

result_status = solver.Solve()

print('Number of variables =', solver.NumVariables())
print('Number of constraints =', solver.NumConstraints())

  # The objective value of the solution.
print('Optimal objective value = %d' % solver.Objective().Value())

B = W

for i in range(len(W)):
    for j in range(len(W[i])):
        # B[i][j] = B_ij
        B[i][j] = eval("B_" + str(i) + str(j) + ".solution_value()")


