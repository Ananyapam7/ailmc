"""
Adult with counterfactual fairness constraint (logistic)

@author: chamon
"""

import os
import numpy as np

import jax.numpy as jnp
from jax import device_put

from data import datasets

CWD = os.path.abspath("")


def preprocess(a):
    # Load Adult data
    preprocess_funcs = [
        datasets.Drop(
            ["fnlwgt", "educational-num", "relationship", "capital-gain", "capital-loss"]
        ),
        datasets.Recode(
            "education",
            {
                "<= K-12": [
                    "Preschool",
                    "1st-4th",
                    "5th-6th",
                    "7th-8th",
                    "9th",
                    "10th",
                    "11th",
                    "12th",
                ]
            },
        ),
        datasets.Recode("race", {"Other": ["Other", "Amer-Indian-Eskimo"]}),
        datasets.Recode(
            "marital-status",
            {
                "Married": ["Married-civ-spouse", "Married-AF-spouse", "Married-spouse-absent"],
                "Divorced/separated": ["Divorced", "Separated"],
            },
        ),
        datasets.Recode(
            "native-country",
            {
                "South/Central America": [
                    "Columbia",
                    "Cuba",
                    "Guatemala",
                    "Haiti",
                    "Ecuador",
                    "El-Salvador",
                    "Dominican-Republic",
                    "Honduras",
                    "Jamaica",
                    "Nicaragua",
                    "Peru",
                    "Trinadad&Tobago",
                ],
                "Europe": [
                    "England",
                    "France",
                    "Germany",
                    "Greece",
                    "Holand-Netherlands",
                    "Hungary",
                    "Italy",
                    "Ireland",
                    "Portugal",
                    "Scotland",
                    "Poland",
                    "Yugoslavia",
                ],
                "Southeast Asia": ["Cambodia", "Laos", "Philippines", "Thailand", "Vietnam"],
                "Chinas": ["China", "Hong", "Taiwan"],
                "USA": ["United-States", "Outlying-US(Guam-USVI-etc)", "Puerto-Rico"],
            },
        ),
        datasets.QuantileBinning("age", 6),
        datasets.Binning("hours-per-week", bins=[0, 40, 100]),
        datasets.Dummify(datasets.Adult.categorical + ["age", "hours-per-week"]),
    ]

    result = a
    for f in preprocess_funcs:
        result = f(result)
    return result


def get_data():
    trainset = datasets.Adult(
        root=os.path.join(CWD, "data"),
        train=True,
        target_name="income",
        preprocess=preprocess,
        transform=datasets.ToArray(),
        target_transform=datasets.ToArray(),
    )
    testset = datasets.Adult(
        root=os.path.join(CWD, "data"),
        train=False,
        target_name="income",
        preprocess=preprocess,
        transform=datasets.ToArray(),
        target_transform=datasets.ToArray(),
    )
    fullset = datasets.Adult(
        root=os.path.join(CWD, "data"), train=True, target_name="income", preprocess=preprocess
    )

    var_names = fullset[0][0].columns
    gender_idx = [
        idx for idx, name in enumerate(fullset[0][0].columns) if name.startswith("gender")
    ][0]

    X, y, _ = trainset[np.arange(len(trainset.data))]
    male_idx = np.argwhere(X[:, gender_idx]).flatten()
    female_idx = np.argwhere(1.0 - X[:, gender_idx]).flatten()

    n_data = X.shape[0]
    X = jnp.hstack((jnp.ones((n_data, 1)), X))

    X_test, y_test, _ = testset[np.arange(len(testset.data))]
    test_male_idx = np.argwhere(X_test[:, gender_idx]).flatten()
    test_female_idx = np.argwhere(1.0 - X_test[:, gender_idx]).flatten()

    n_test = X_test.shape[0]
    X_test = jnp.hstack((jnp.ones((n_test, 1)), X_test))

    gender_idx += 1  # due to the additional all-ones column
    return (
        device_put(jnp.array(X)),
        device_put(jnp.array(y)),
        var_names,
        gender_idx,
        male_idx,
        female_idx,
        X_test,
        y_test,
        test_male_idx,
        test_female_idx,
    )
