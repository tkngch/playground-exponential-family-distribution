from unittest import TestCase

from efd.linear_algebra.gaussian_elimination.determinant import determinant
from efd.linear_algebra.matrix import square_matrix


class TestDeterminant(TestCase):
    def test_determinant_case_0(self):
        # Taken from the wikipedia on 11 March 2023.
        # https://en.wikipedia.org/wiki/Determinant#2_%C3%97_2_matrices
        m = square_matrix((3, 7), (1, -4))
        self.assertAlmostEqual(-19, determinant(m))

    def test_determinant_case_1(self):
        # Taken from the wikipedia on 11 March 2023.
        # https://en.wikipedia.org/wiki/Determinant#Example
        m = square_matrix((-2, -1, 2), (2, 1, 4), (-3, 3, -1))
        self.assertAlmostEqual(54, determinant(m))
