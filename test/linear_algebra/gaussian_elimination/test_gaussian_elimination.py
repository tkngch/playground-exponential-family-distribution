from unittest import TestCase

from efd.linear_algebra.gaussian_elimination.gaussian_elimination import solve
from efd.linear_algebra.matrix import column_vector, square_matrix


class TestGaussianElimination(TestCase):
    def test_solve_case_0(self):
        a = square_matrix((2, 3), (1, -1))
        y = column_vector(6, 0.5)
        solution = column_vector(1.5, 1)
        self.assertEqual(solution, solve(a, y)[0])

    def test_solve_case_1(self):
        a = square_matrix((2, 1, -1), (-3, -1, 2), (-2, 1, 2))
        y = column_vector(8, -11, -3)
        solution = column_vector(2, 3, -1)
        self.assertEqual(solution, solve(a, y)[0])

    def test_solve_case_2(self):
        a = square_matrix((9, 3, 4), (4, 3, 4), (1, 1, 1))
        y = column_vector(7, 8, 3)
        solution = column_vector(-0.2, 4, -0.8)
        self.assertEqual(solution, round(solve(a, y)[0], 4))
