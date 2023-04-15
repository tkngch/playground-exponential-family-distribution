from unittest import TestCase

from efd.linear_algebra.gaussian_elimination.invert import invert
from efd.linear_algebra.matrix import identity_matrix, square_matrix


class TestInvert(TestCase):
    def test_invert_case_0(self):
        m = square_matrix(
            (3, 0, 2),
            (2, 0, -2),
            (0, 1, 1),
        )
        inverse = invert(m)
        expected = square_matrix((0.2, 0.2, 0), (-0.2, 0.3, 1), (0.2, -0.3, 0))
        self.assertEqual(expected, round(inverse, 4))
        self.assertEqual(identity_matrix(3), round(inverse * m, 4))

    def test_invert_case_1(self):
        m = square_matrix(
            (5, 4, 3, 2, 1),
            (4, 3, 2, 1, 5),
            (3, 2, 9, 5, 4),
            (2, 1, 5, 4, 3),
            (1, 2, 3, 4, 5),
        )
        inverse = invert(m)
        self.assertEqual(identity_matrix(5), round(inverse * m, 4))
