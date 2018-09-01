#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 10:02:11 2018

max  24*B_00 + 40*B_01 + 79*B_02 + 23*B_10 + 48*B_11
s.t. 
      11*B_00 + 22*B_01 + 33*B_02 + 10*B_10 + 23*B_11 <= 37
      B_00 + B_01 + B_02 <= 1
      B_10 + B_11 <= 1
      B_00, B_01, B_02, B_10, B_11 are binary
"""

# w = [[11, 22, 33], [10, 23]]
# v = [[24, 40, 79], [23, 48]]
# W = 37


from ortools.linear_solver import pywraplp

solver = pywraplp.Solver('SolveIntegerProblem',
                         pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

b_00 = solver.IntVar(0.0, 1.0, 'b_00')
b_01 = solver.IntVar(0.0, 1.0, 'b_01')
b_02 = solver.IntVar(0.0, 1.0, 'b_02')
b_10 = solver.IntVar(0.0, 1.0, 'b_10')
b_11 = solver.IntVar(0.0, 1.0, 'b_11')


constraint1 = solver.Constraint(-solver.infinity(), 37)
constraint1.SetCoefficient(b_00, 11)
constraint1.SetCoefficient(b_01, 22)
constraint1.SetCoefficient(b_02, 33)
constraint1.SetCoefficient(b_10, 10)
constraint1.SetCoefficient(b_11, 23)


constraint2 = solver.Constraint(1, 1)
constraint2.SetCoefficient(b_00, 1)
constraint2.SetCoefficient(b_01, 1)
constraint2.SetCoefficient(b_02, 1)
constraint2.SetCoefficient(b_10, 0)
constraint2.SetCoefficient(b_11, 0)


constraint3 = solver.Constraint(1, 1)
constraint3.SetCoefficient(b_00, 0)
constraint3.SetCoefficient(b_01, 0)
constraint3.SetCoefficient(b_02, 0)
constraint3.SetCoefficient(b_10, 1)
constraint3.SetCoefficient(b_11, 1)


objective = solver.Objective()
objective.SetCoefficient(b_00, 24)
objective.SetCoefficient(b_01, 40)
objective.SetCoefficient(b_02, 79)
objective.SetCoefficient(b_10, 23)
objective.SetCoefficient(b_11, 48)
objective.SetMaximization()

result_status = solver.Solve()

assert result_status == pywraplp.Solver.OPTIMAL
assert solver.VerifySolution(1e-7, True)

print('Number of variables =', solver.NumVariables())
print('Number of constraints =', solver.NumConstraints())
print('Optimal objective value = %d' % solver.Objective().Value())
