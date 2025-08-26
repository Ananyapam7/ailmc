# Augmented Lagrangian Langevin Monte Carlo (AL-LMC)

A JAX implementation of constrained sampling algorithms using Langevin Monte Carlo with augmented Lagrangian methods for Bayesian inference with fairness constraints.

## Overview

This project implements three main sampling algorithms:

1. **Standard Langevin Monte Carlo (LMC)** - Unconstrained sampling
2. **Primal-Dual Langevin Monte Carlo (PD-LMC)** - Constrained sampling using primal-dual dynamics
3. **Augmented Lagrangian Langevin Monte Carlo (AL-LMC)** - Constrained sampling with augmented Lagrangian penalty terms

The algorithms are designed for Bayesian inference problems with equality and inequality constraints, with a focus on fairness-aware machine learning applications.

## Algorithms

### AL-LMC Algorithm
The augmented Lagrangian potential function is:
```
U(x, λ, ν) = f(x) + λᵀg(x) + νᵀh(x) + (ρ/2)||g(x)||² + (ρ/2)||h(x)||²
```

Where:
- `f(x)`: Negative log posterior (potential function)
- `g(x)`: Inequality constraints
- `h(x)`: Equality constraints  
- `λ, ν`: Dual variables
- `ρ`: Penalty parameter

The algorithm alternates between:
1. Langevin sampling from the augmented potential
2. Dual variable updates using constraint violations
3. Constraint tracker updates with exponential averaging

## Experiments

The main experiment demonstrates **Bayesian logistic regression with fairness constraints** using the Adult dataset:

- **Fairness constraint**: Demographic parity between gender groups
- **Comparison**: Standard LMC vs PD-LMC vs AL-LMC
- **Analysis**: Constraint satisfaction, dual variable convergence, feature importance

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from pdlmc import allmclc_run_chain

# Run AL-LMC sampling
final_key, trajectory = allmclc_run_chain(
    initial_key=key,
    f=neglogposterior,      # Potential function
    g=inequality_constraints,
    h=equality_constraints,
    iterations=1000,
    lmc_steps=10,
    step_size_x=1e-4,
    step_size_lmbda=5e-3,
    penalty_rho=1.0,
    tracker_rate=0.1,
    # ... other parameters
)
```

## Required Packages

- `jax` - Core computation framework
- `numpy` - Numerical operations
- `matplotlib` - Plotting and visualization
- `scikit-learn` - Data preprocessing
- `pandas` - Data manipulation