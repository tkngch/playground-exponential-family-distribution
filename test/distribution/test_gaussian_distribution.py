from math import exp, pi
from unittest import TestCase

from efd.distribution.gaussian_distribution import GaussianDistribution
from efd.linear_algebra import (
    ColumnVector,
    SquareMatrix,
    as_scalar,
    column_vector,
    identity_matrix,
    square_matrix,
)
from efd.linear_algebra.gaussian_elimination import determinant, invert


class TestGaussianDistribution(TestCase):
    def test_density_univariate_case_0(self):
        x = column_vector(0)
        mean = column_vector(0)
        covariance = identity_matrix(1)

        distribution = GaussianDistribution.of(mean, covariance)
        # The expected result is taken from Wolfram Alpha.
        self.assertAlmostEqual(0.398942, distribution.density(x), places=6)

    def test_density_univariate_case_1(self):
        x = column_vector(1)
        mean = column_vector(-1)
        covariance = 2 * identity_matrix(1)

        distribution = GaussianDistribution.of(mean, covariance)
        self.assertAlmostEqual(
            _reference_density(x, mean, covariance), distribution.density(x)
        )

    def test_density_bivariate_case_0(self):
        x = column_vector(0, 0)
        mean = column_vector(0, 0)
        covariance = identity_matrix(2)

        distribution = GaussianDistribution.of(mean, covariance)
        self.assertAlmostEqual(
            _reference_density(x, mean, covariance), distribution.density(x)
        )

    def test_density_bivariate_case_1(self):
        x = column_vector(0.2, -0.3)
        mean = column_vector(0.1, 10)
        covariance = square_matrix((1.5, 0.1), (0.1, 2.1))

        distribution = GaussianDistribution.of(mean, covariance)
        self.assertAlmostEqual(
            _reference_density(x, mean, covariance), distribution.density(x)
        )

    def test_density_trivariate_case_0(self):
        x = column_vector(0, 0, 0)
        mean = column_vector(0, 0, 0)
        covariance = identity_matrix(3)

        distribution = GaussianDistribution.of(mean, covariance)
        self.assertAlmostEqual(
            _reference_density(x, mean, covariance), distribution.density(x)
        )

    def test_density_trivariate_case_1(self):
        x = column_vector(1, 2, 3)
        mean = column_vector(-3, -2, -1)
        covariance = square_matrix((3, 2, 1), (2, 3, 1), (1, 1, 3))

        distribution = GaussianDistribution.of(mean, covariance)
        self.assertAlmostEqual(
            _reference_density(x, mean, covariance), distribution.density(x)
        )


def _reference_density(
    x: ColumnVector, mean: ColumnVector, covariance: SquareMatrix
) -> float:
    if x.n_rows != mean.n_rows:
        raise ValueError(
            f"Dimension mismatch: x has {x.n_rows} rows "
            f"but the mean has {mean.n_rows} rows."
        )
    if mean.n_rows != covariance.n_rows:
        raise ValueError(
            f"Dimension mismatch: the mean has {mean.n_rows} rows "
            f"but the covariance has {covariance.n_rows}."
        )

    return (
        ((2 * pi) ** (-x.n_rows / 2))
        * (determinant(covariance) ** (-1 / 2))
        * exp(
            (-1 / 2)
            * as_scalar((x - mean).transpose() * invert(covariance) * (x - mean))
        )
    )
