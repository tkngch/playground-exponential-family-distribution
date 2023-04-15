from __future__ import annotations

import random
from dataclasses import dataclass
from functools import cached_property, lru_cache
from itertools import chain
from math import log, pi
from typing import Iterator

from ..linear_algebra import (
    ColumnVector,
    Matrix,
    SquareMatrix,
    as_scalar,
    column_vector,
)
from ..linear_algebra.gaussian_elimination import determinant, invert
from .exponential_family import ExponentialFamily, Parameter


@dataclass(frozen=True)
class GaussianDistributionParameter(Parameter):
    mean: ColumnVector
    covariance: SquareMatrix

    @cached_property
    def precision(self) -> SquareMatrix:
        return invert(self.covariance)


class GaussianDistribution(ExponentialFamily):
    @staticmethod
    def of(mean: ColumnVector, covariance: SquareMatrix) -> GaussianDistribution:
        if mean.n_rows != covariance.n_rows:
            raise ValueError(
                f"Mismatching dimensions: mu has {mean.n_rows} elements, "
                f"while sigma is {covariance.n_rows}-by-{covariance.n_columns}."
            )

        return GaussianDistribution(GaussianDistributionParameter(mean, covariance))

    def __init__(self, parameter: GaussianDistributionParameter):
        self.__parameter = parameter
        self.__random_number_generator = random.Random()

    @property
    def parameter(self) -> GaussianDistributionParameter:
        return self.__parameter

    @cached_property
    def natural_parameter(self) -> ColumnVector:
        return column_vector(*_natural_parameter_elements(self.parameter))

    @staticmethod
    def sufficient_statistics(x: ColumnVector) -> ColumnVector:
        return column_vector(*_sufficient_statistics_elements(x))

    @staticmethod
    def base_measure(x: ColumnVector) -> float:
        return pow(2 * pi, -x.n_rows / 2)

    @staticmethod
    @lru_cache
    def log_partition(theta: GaussianDistributionParameter) -> float:
        return (
            as_scalar(theta.mean.transpose() * theta.precision * theta.mean)
            + log(determinant(theta.covariance))
        ) / 2


def _natural_parameter_elements(
    theta: GaussianDistributionParameter,
) -> Iterator[float]:
    return chain.from_iterable(
        (_flatten(theta.precision * theta.mean), _flatten(-0.5 * theta.precision))
    )


def _sufficient_statistics_elements(x: ColumnVector) -> Iterator[float]:
    return chain.from_iterable((_flatten(x), _flatten(x * x.transpose())))


def _flatten(m: Matrix) -> Iterator[float]:
    for row_index in range(m.n_rows):
        for column_index in range(m.n_columns):
            yield m.rows[row_index][column_index]
