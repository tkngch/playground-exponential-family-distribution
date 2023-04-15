from unittest import TestCase, main

from efd.linear_algebra.matrix import matrix


class TestMatrix(TestCase):
    def test_shape(self):
        m = matrix((1, 2), (3, 4), (5, 6))
        self.assertEqual(3, m.n_rows)
        self.assertEqual(2, m.n_columns)

    def test_add_scalar(self):
        original = matrix((1, 2), (3, 4), (5, 6))
        added_one = matrix((2, 3), (4, 5), (6, 7))
        self.assertEqual(added_one, 1 + original)

    def test_add_matrix(self):
        original = matrix((1, 2), (3, 4), (5, 6))
        to_add = matrix((2, 1), (4, 3), (6, 5))
        added = matrix((3, 3), (7, 7), (11, 11))
        self.assertEqual(added, original + to_add)
        self.assertEqual(added, to_add + original)

    def test_multiply_scalar(self):
        original = matrix((1, 2), (3, 4), (5, 6))
        doubled = matrix((2, 4), (6, 8), (10, 12))
        self.assertEqual(doubled, 2 * original)

    def test_multiply_matrix(self):
        original = matrix((1, 2), (3, 4), (5, 6))
        to_multiply = matrix((2, 1, 0), (3, 4, 5))
        multiplied = matrix((8, 9, 10), (18, 19, 20), (28, 29, 30))
        self.assertEqual(multiplied, original * to_multiply)

    def test_transpose(self):
        m = matrix((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        transposed = matrix((1.0, 3.0, 5.0), (2.0, 4.0, 6.0))
        self.assertEqual(transposed, m.transpose())


if __name__ == "__main__":
    main()
