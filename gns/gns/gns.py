#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:10:23 2018

@author: matthew.mu
"""


from ortools.linear_solver import pywraplp
import warnings



class GNS(object):
    
    def __init__(self, values, weights, capacity, style='hard'):
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.style=style
        try:
            self.solver = _new_solver('gns', integer=True)
        except:
            warnings.warn("CPLEX is not found and switch to CBC!")
            self.solver = _new_solver('gns', integer=True, cplex=False)
        
    def solve(self):
        
        W, V, capacity, solver, style \
            = self.weights, self.values, self.capacity, self.solver, self.style
        
        # create variables
        B = [[solver.IntVar(0, solver.infinity(), f'B_{i}_{j}')  
                for j in range(len(W[i]))] 
                    for i in range(len(W))]

        # constraints
        if style=='hard':
            for i in range(len(W)):
                solver.Add(sum(B[i][j] for j in range(len(W[i])))==1)
        if style=='soft':
            for i in range(len(W)):
                solver.Add(sum(B[i][j] for j in range(len(W[i])))<=1)
                
        solver.Add(sum(sum(B[i][j]*W[i][j] for j in range(len(W[i]))) for i in range(len(W)))<=capacity)
      
        # obj.
        Value = sum(sum(B[i][j]*V[i][j] for j in range(len(W[i]))) for i in range(len(W)))
        solver.Maximize(Value)

        # solve
        solver.Solve()
        
        # collect solution
        self.num_variables = solver.NumVariables()
        self.num_contraints = solver.NumConstraints()

        self.obj_value = _obj_val(solver)
        self.B = _sol_val(B) # solution value
        

# utility functions
def _new_solver(name, integer=False, cplex=True):
  return pywraplp.Solver(name,
    pywraplp.Solver.CPLEX_MIXED_INTEGER_PROGRAMMING
      if integer and cplex else pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
      if integer else pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

def _sol_val(x):
  if type(x) is not list:
    return 0 if x is None \
             else x if isinstance(x,(int,float)) \
                    else x.SolutionValue() if x.Integer() is False \
                                           else int(x.SolutionValue())
  elif type(x) is list:
    return [_sol_val(e) for e in x]

def _obj_val(x):
  return x.Objective().Value()