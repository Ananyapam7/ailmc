"""
Bayesian logistic regression with fairness constraint
"""

import os
import sys

from jax import jit
from jax.nn import sigmoid
from jax.tree_util import Partial
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np

from data.adult import get_data

CWD = os.path.dirname(__file__)
sys.path.append(os.path.join(CWD, ".."))
from pdlmc import pdlmc_run_chain


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
def _ineqconst(beta, delta):
    return jnp.array(
        [
            100 * (sigmoid(jnp.dot(X, beta)).mean() - sigmoid(jnp.dot(X[male_idx, :], beta)).mean())
            - 1,
            100
            * (sigmoid(jnp.dot(X, beta)).mean() - sigmoid(jnp.dot(X[female_idx, :], beta)).mean())
            - delta,
        ]
    )


# PD-LMC
init_beta = jr.normal(INITKEY, (X.shape[1],)) * 0.1
pdlmc_traj = {}
for d in [1, 2, 3, 4, 5]:
    ineqconst = Partial(_ineqconst, delta=d)
    _, pdlmc_traj[d] = pdlmc_run_chain(
        jr.PRNGKey(1234),
        neglogposterior,
        ineqconst,
        lambda _: 0,
        ITERATIONS,
        1,
        0,
        1e-4,
        5e-3,
        0,
        init_beta,
        jnp.zeros(2),
        jnp.array(0.0),
    )

np.savez(
    os.path.join(CWD, "fairness_relaxed"),
    pdlmc_x_1=pdlmc_traj[1].x,
    pdlmc_lambda_1=pdlmc_traj[1].lmbda,
    pdlmc_x_2=pdlmc_traj[2].x,
    pdlmc_lambda_2=pdlmc_traj[2].lmbda,
    pdlmc_x_3=pdlmc_traj[3].x,
    pdlmc_lambda_3=pdlmc_traj[3].lmbda,
    pdlmc_x_4=pdlmc_traj[4].x,
    pdlmc_lambda_4=pdlmc_traj[4].lmbda,
    pdlmc_x_5=pdlmc_traj[5].x,
    pdlmc_lambda_5=pdlmc_traj[5].lmbda,
)
