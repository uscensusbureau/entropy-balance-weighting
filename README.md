Accelerated Entropy Balancing Sample Weighting
------------------------------------
This package implements an Entropy Balancing Weighting (EBW) routine for dense or sparse matrices.
The goal of this package is to scale the EBW algorithm to allow it to work with large, complicated data
sets. A particular focus is then on allowing sparse inputs, collinearity, and potential 
infeasibility without additional front-end data manipulation by the user.

Scaling up EBW problems is largely possible due to the sparsity of naturally-occurring data, e.g. models with group-specific moments. Utilizing sparsity of datasets, models with millions of observations and hundreds of thousands of constraints have successfully converged in minutes. In addition, extensions like bound constraints and handling collinear and inconsistent moments have been added to the standard approach.

The paper detailing the algorithm is forthcoming.

Requirements:
-------------
- Python 3.9+
- Required packages (tested versions listed, should work on most others):
  - numpy (2.2, 1.26)
  - scipy (1.15.1, 1.13.1)
  - pypardiso (0.4.6)
  - sparse_dot_mkl (0.9.7)
  - numexpr (2.10.2)
- Required packages for testing:
  - polars (1.0+)
  - pytest (8.3.0+)
- Required packages for examples:
  - requests

Installation:
-------------
Once the release `.whl` file is downloaded and stored in \<wheel location\>, a simple
```pip install <wheel location>```
should work.

Problem and Usage:
-------------------
The primary function in this package is ```entropy_balance_weighting.entropy_balance```, which solves this problem for any consistent set of constraints that can be satisfied with positive weights.

The EBW problem solved here considers a set of $N$ sampled units, with initial unit weights $w_0$, an $N\times K$ matrix of observable characteristics at the unit level $X$, and a $K\times 1$ vector of known external unit-level mean "targets" $m$ that the reweighted moments should match. Distance between the new weights and old weights is based on the Kullback-Leibler divergence and can be written in terms of the ratio $r \equiv w_{1}/w_{0}$ between the new weights and the old weights:

$$\phi(r; w_0)=\sum w_{0,i} (\log(r_{i}) - r_{i} + 1).$$

The optimization problem can then be written

$$\min_{r\in \mathbb{R}^N} \phi(r; w_0)$$


$$\text{s.t. } (\text{Diag}({w_{i,0})}X)^T r = m \cdot \sum_{i=1}^N w_{i,0}, \hspace{3pt} r\geq 0.$$


The ```entropy_balance``` function takes three required keyword arguments: ```x_sample```, the $N\times K$ array dense or sparse matrix (numpy.array or scipy.sparse.csc_array/csr_array) of unit-level observations; ```weights0```, a numpy vector of the initial weights, best scaled to have mean 1; and ```mean_population_moments```, a numpy vector of the externally-known unit-level mean of each column of $X$. As written above, the algorithm matches the aggregate moments, so keeping the relative scaling of $X$, $m$, and $w_{i,0}$ consistent with the formulation above makes the algorithm more stable, even if not theoretically necessary. 

The function returns an ```EntropyBalanceResults``` object, which has a property ```EntropyBalanceResults.new_weights```, the new weights that match the moments (if possible) while being minimum distance from the initial weights.

Passing in optional arguments to ```entropy_balance``` can allow for setting bounds in the resulting ratios and dealing with infeasibility of the linear system; see the section "Bounded and Infeasible" below.

Treatment/Control Weighting:
---------
This package can also be used to reweight subsets of observations to all have the same weighted mean characteristics over multiple dimensions. See the ```examples/pums_example.py``` file for an example of how to write the moments as a special case of the more general problem above. In that example, data from the American Community Survey is reweighted at the state level so all states have equal post-weighting means of a number of household-level variables.

```entropy_balance``` Example:
--------

```
>>> import numpy as np
>>> import entropy_balance_weighting as ebw
>>> import scipy.sparse as sp
>>> 
>>> sample_size = 5
>>> x = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
>>> mean_population_moments = np.array([.35, .30, .33])
>>> weights0 = np.ones(sample_size)
>>> x.T.dot(weights0) - sample_size * mean_population_moments  # Calculate miss in weighted aggregate moments.
array([-0.75,  0.5 ,  0.35])
>>>
>>> res_ebw = ebw.entropy_balance(x_sample = x, weights0 = weights0, mean_population_moments = mean_population_moments)
>>> res_ebw.new_weights
array([1.75 , 0.75 , 0.75 , 0.825, 0.825])
>>> x.T.dot(res_ebw.new_weights) - sample_size * mean_population_moments # Aggregate weighted moments now matched exactly.
array([6.72795153e-14, 1.28785871e-14, 4.66293670e-15])
>>>
>>> x_sparse = sp.csr_array(x) # Can use sparse matrices for memory/computation savings
>>> res_sparse = ebw.entropy_balance(x_sample = x_sparse, weights0 = weights0, mean_population_moments = mean_population_moments)
>>> assert np.allclose(res_ebw.new_weights, res_sparse.new_weights)
```

More Examples:
--------------
More detailed examples can be found in the `./examples` directory in the main repository. In particular, the `pums_example.py` gives an example
of downloading the US Census 2023 1-Year American Community Survey and reweighting to exactly match the state-level number of housing units from the US Census Population Estimates program.

Bounded and Infeasible:
-----------------------
Passing the kwarg ```options={"bounds": (lbound, ubound)}``` for scalar `lbound` and `ubound` will solve the problem above with $\text{lbound} \leq r_i \leq \text{ubound }\forall i$. For no upper bound, pass `None` for `ubound`. 

There is a tradeoff between bound constraints and feasibility of the linear system matching the moments. To make bounds generally useful, the bounded `entropy_balance` algorithm runs "elastic mode", a variant problem that has the following properties:

1. If the bounded problem is feasible, `entropy_balance(..., options={"bounds": ...})` returns the feasible solution.
2. If the problem is infeasible, `entropy_balance` returns a point $r^\star$ that is feasible with respect to the bounds. This $r^\star$ minimizes the $L^1$ norm of the moment violations over the other feasible $r$ that satisfy $\phi(r) \leq \phi(r^\star)$; that is, the only way to make the moments match any better would be to increase the criterion. In practice, this matches as many moments exactly as possible, making it possible to identify which moments are impossible to simultaneously match.

The bounded problem is necessarily slower than the unbounded problem, but in theory should converge in every run. After running with bounds, the ```EntropyBalanceResult.constraint_violations``` property can be used to judge if the original problem was feasible or not. The feasibility of the unconstrained problem can be checked by passing `{"bounds": (0, None)}`.

Bounds and Elastic Mode Example:
--------------------------------
```
<continued from above>
>>> res_bounded= ebw.entropy_balance(x_sample = x, weights0 = weights0, mean_population_moments = mean_population_moments, options={"bounds": (0.5, 1.5)})
>>> res_bounded.new_weights
array([1.5 , 0.75 , 0.75 , 0.825, 0.825])
>>> res_bounded.constraint_violations # New bounds make problem infeasible, so return new weights that give "best fit" of moments.
array([-2.50000000e-01, -8.58102478e-11, -5.19522203e-11])

```



Penalty Formulation:
-------------------
A secondary, currently slightly less supported, function in this package is ```entropy_balance_weighting.entropy_balance_penalty```, which solves the problem 

$$\min_{r\in \mathbb{R}^N} \phi(r; w_0) + \frac{1}{2} (A^T r-m)^T \text{Diag}(p_k)(A^T r - m)$$
$$\text{s.t. }  A=\text{Diag}(\frac{w_{i,0}}{\sum_j w_{j,0}})X, \hspace{3pt} r\geq 0.$$

where $p_{k}$ is a vector of positive penalty parameters chosen by the user. As above, this method also supports additional bound constraints on $r$ if provided.


This version of the problem moves the constraints into a smooth penalty function, which ensures feasibility. 

The advantage of this method is allowing for infeasible and collinear moments without any complications, and the simple form of the problem makes optimization fast. 

The downside compared to the above methods is that for any finite $p$ the solution will not match the solution of the original EBW problem exactly, and the choice of $p$ is arbitrary for the end user. Additionally, unlike the bounded elastic mode version above, it is not possible to algorithmically use this version of the problem to determine the feasibility of the original problem. 

Issues:
-------
We appreciate any feedback you would like to provide us; please post any questions that you may have in the GitHub issues section.

Citation Information:
---------------------
Please cite this package in any work where it proves useful.
```
@software{Sanders_Accelerated_Entropy_Balance_2025,
author = {Sanders, Carl},
month = mar,
title = {{Accelerated Entropy Balance Sample Weighting}},
url = {https://github.com/uscensusbureau/entropy-balance-weighting},
version = {0.5.0},
year = {2025}
}
```

Disclaimers:
------------
This work was performed as part of the [National Experimental Wellbeing Statistics](https://census.gov/data/experimental-data-products/national-experimental-wellbeing-statistics.html) project at the U.S. Census Bureau.
The views expressed are those of the authors and not those of the U.S. Census Bureau. The U.S. Census Bureau has reviewed this data product to ensure appropriate access, use, and disclosure avoidance protection of the confidential source data used to produce this product (Data Management System (DMS) number: P-6000725; Disclosure Review Board (DRB) approval number: CBDRB-FY25-SEHSD003-037).

