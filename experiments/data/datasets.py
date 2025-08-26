# -*- coding: utf-8 -*-
"""
Datasets

@author: chamon
"""

import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


import jax.numpy as jnp
import pandas as pd


class Adult:
    # Categorical variables
    categorical = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "native-country",
        "income",
    ]

    def __init__(
        self, root, target_name, train=True, preprocess=None, transform=None, target_transform=None
    ):
        self.classes = ("<= 50k", "> 50k")
        self.train = train

        # Read CSV file
        column_names = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "educational-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "gender",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income",
        ]

        # Load data
        if self.train:
            self.data = pd.read_csv(
                os.path.join(root, "adult.data"),
                sep=r",\s",
                header=None,
                names=column_names,
                engine="python",
            )
        else:
            self.data = pd.read_csv(
                os.path.join(root, "adult.test"),
                sep=r",\s",
                header=None,
                names=column_names,
                engine="python",
            )
            self.data["income"] = self.data["income"].replace(r"\.", r"", regex=True)

        # Declare categorical variables
        for var_name in Adult.categorical:
            self.data[var_name] = self.data[var_name].astype("category")

        if preprocess is not None:
            self.data = preprocess(self.data)

        # Recover response variable
        self.target = self.data.filter(regex=f"^{target_name}", axis=1)
        self.data = self.data.drop(self.target.columns, axis=1)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if type(index) is int:
            data, target = self.data.loc[[index]], self.target.loc[[index]]
        else:
            data, target = self.data.loc[index], self.target.loc[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target, index

    def __len__(self):
        return self.target.shape[0]


# Dataset transformations
class Drop:
    def __init__(self, var_names):
        self.var_names = var_names

    def __call__(self, sample):
        return sample.drop(self.var_names, axis=1)


class Recode:
    def __init__(self, var_name, dictionary):
        self.var_name = var_name
        self.dictionary = dictionary

    def __call__(self, sample):
        transposed_dicitionary = {}
        for new_value, old_values in self.dictionary.items():
            for value in old_values:
                transposed_dicitionary[value] = new_value

        if isinstance(sample[self.var_name].dtype, pd.CategoricalDtype):
            sample[self.var_name] = (
                sample[self.var_name]
                .astype("object")
                .replace(transposed_dicitionary)
                .astype("category")
            )
        else:
            sample = sample.replace({self.var_name: transposed_dicitionary})

        return sample


class Dummify:
    def __init__(self, var_names):
        self.var_names = var_names

    def __call__(self, sample):
        for name in self.var_names:
            if name in sample.columns:
                if len(sample[name].cat.categories) > 2:
                    sample = pd.get_dummies(sample, prefix=[name], columns=[name])
                else:
                    sample = pd.get_dummies(sample, prefix=[name], columns=[name], drop_first=True)
        return sample


class QuantileBinning:
    def __init__(self, var_name, quantile):
        self.var_name = var_name
        self.quantile = quantile

    def __call__(self, sample):
        sample[self.var_name] = pd.qcut(sample[self.var_name], q=self.quantile)

        return sample


class Binning:
    def __init__(self, var_name, bins):
        self.var_name = var_name
        self.bins = bins

    def __call__(self, sample):
        sample[self.var_name] = pd.cut(sample[self.var_name], bins=self.bins, include_lowest=True)

        return sample


class ToArray:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, sample):
        return jnp.array(sample.values, dtype=jnp.float32, **self.kwargs).squeeze()
