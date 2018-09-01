Generalized Knapsack Problem (gns)
================
Cun Mu
August 16, 2018

Problem Statement
-----------------

The package aims to solve the following optimization problems:

*m**a**x* ∑<sub>*i*</sub>∑<sub>*j*</sub>*V*<sub>*i**j*</sub>*B*<sub>*i**j*</sub> *s*.*t*. s.t. ∑<sub>*i*</sub>∑<sub>*j*</sub>*W*<sub>*i**j*</sub>*B*<sub>*i**j*</sub> ≤ *W*,  ∑<sub>*j*</sub>*B*<sub>*i**j*</sub> = 1 ∀*i*,  *B*<sub>*i**j*</sub> ∈ {0, 1} ∀*i*, *j*

*m**a**x* ∑<sub>*i*</sub>∑<sub>*j*</sub>*V*<sub>*i**j*</sub>*B*<sub>*i**j*</sub> *s*.*t*. s.t. ∑<sub>*i*</sub>∑<sub>*j*</sub>*W*<sub>*i**j*</sub>*B*<sub>*i**j*</sub> ≤ *W*,  ∑<sub>*j*</sub>*B*<sub>*i**j*</sub> ≤ 1 ∀*i*,  *B*<sub>*i**j*</sub> ∈ {0, 1} ∀*i*, *j*

When *i=1*, the second problem reduces to standard Knapsack problem. 