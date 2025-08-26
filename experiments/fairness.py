"""
Bayesian logistic regression with fairness constraint
"""

import os
import sys

from jax import jit
from jax.nn import sigmoid
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np

from data.adult import get_data

CWD = os.path.dirname(__file__)
sys.path.append(os.path.join(CWD, ".."))
from pdlmc import lmc_run_chain, pdlmc_run_chain


KEY = jr.PRNGKey(1234)
KEY, INITKEY = jr.split(KEY)
ITERATIONS = int(2e4)

# Load data
(
    X,
    y,
    var_names,
    gender_idx,
    male_idx,
    female_idx,
    X_test,
    y_test,
    test_male_idx,
    test_female_idx,
) = get_data()

X_test_cf = X_test.at[:, gender_idx].set(1 - X_test[:, gender_idx])


# Constraints
@jit
def loglikelihood(beta):
    return jnp.sum(-jnp.log(1 + jnp.exp(-(2 * y - 1) * jnp.dot(X, beta))))


@jit
def logprior(beta):
    return jsp.stats.norm.logpdf(beta[0], loc=0, scale=3) + jnp.sum(
        jsp.stats.norm.logpdf(beta[1:], loc=0, scale=3)
    )


@jit
def neglogposterior(beta):
    return -loglikelihood(beta) - logprior(beta)


@jit
def ineqconst(beta):
    return jnp.array(
        [
            100 * (sigmoid(jnp.dot(X, beta)).mean() - sigmoid(jnp.dot(X[male_idx, :], beta)).mean())
            - 1,
            100
            * (sigmoid(jnp.dot(X, beta)).mean() - sigmoid(jnp.dot(X[female_idx, :], beta)).mean())
            - 1,
        ]
    )


# LMC
init_beta = jr.normal(INITKEY, (X.shape[1],)) * 0.1
lmc_traj = lmc_run_chain(KEY, neglogposterior, ITERATIONS, 1e-4, init_beta)

# PD-LMC
KEY, pdlmc_traj = pdlmc_run_chain(
    initial_key=KEY,
    f=neglogposterior,
    g=ineqconst,
    h=lambda _: 0,
    iterations=ITERATIONS,
    lmc_steps=1,
    burnin=0,
    step_size_x=1e-4,
    step_size_lmbda=5e-3,
    step_size_nu=0,
    initial_x=init_beta,
    initial_lmbda=jnp.zeros(2),
    initial_nu=jnp.array(0.0),
)

np.savez(
    os.path.join(CWD, "fairness"),
    pdlmc_x=pdlmc_traj.x,
    pdlmc_lambda=pdlmc_traj.lmbda,
    pdlmc_nu=pdlmc_traj.nu,
    lmc_x=lmc_traj.x,
)
