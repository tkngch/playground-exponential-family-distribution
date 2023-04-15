from unittest import TestCase

from efd.linear_algebra.gaussian_elimination.elementary_row_operator import (
    ElementaryAxpy,
    ElementaryMultiplication,
    ElementaryPermutation,
    apply_operators,
)
from efd.linear_algebra.matrix import square_matrix


class TestElementaryPermutation(TestCase):
    def test_noop(self):
        m = square_matrix((1, 2, 3), (4, 5, 6), (7, 8, 9))
        operator = ElementaryPermutation.noop()
        self.assertEqual(m, operator.apply(m))

    def test_noop_determinant(self):
        operator = ElementaryPermutation.noop()
        self.assertEqual(1.0, operator.matrix_determinant)

    def test_apply(self):
        m = square_matrix((1, 2, 3), (4, 5, 6), (7, 8, 9))
        operator = ElementaryPermutation((0, 1))
        expected = square_matrix((4, 5, 6), (1, 2, 3), (7, 8, 9))
        self.assertEqual(expected, operator.apply(m))

    def test_inverted(self):
        m = square_matrix((1, 2, 3), (4, 5, 6), (7, 8, 9))
        operator = ElementaryPermutation((0, 1))
        result = apply_operators(m, (operator, operator.inverted()))
        self.assertEqual(m, result)


class TestElementaryMultiplication(TestCase):
    def test_apply(self):
        m = square_matrix((1, 2, 3), (4, 5, 6), (7, 8, 9))
        operator = ElementaryMultiplication(1, 2.0)
        expected = square_matrix((1, 2, 3), (8, 10, 12), (7, 8, 9))
        self.assertEqual(expected, operator.apply(m))

    def test_inverted(self):
        m = square_matrix((1, 2, 3), (4, 5, 6), (7, 8, 9))
        operator = ElementaryMultiplication(1, 2.0)
        result = apply_operators(m, (operator, operator.inverted()))
        self.assertEqual(m, result)


class TestElementaryAxpy(TestCase):
    def test_apply(self):
        m = square_matrix((1, 2, 3), (4, 5, 6), (7, 8, 9))
        operator = ElementaryAxpy(2.0, 0, 1)
        expected = square_matrix((1, 2, 3), (6, 9, 12), (7, 8, 9))
        self.assertEqual(expected, operator.apply(m))

    def test_inverted(self):
        m = square_matrix((1, 2, 3), (4, 5, 6), (7, 8, 9))
        operator = ElementaryAxpy(2.0, 0, 1)
        result = apply_operators(m, (operator, operator.inverted()))
        self.assertEqual(m, result)
