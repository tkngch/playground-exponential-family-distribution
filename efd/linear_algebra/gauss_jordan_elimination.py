import dataclasses
from typing import Iterator, Tuple

from .matrix import (
    ColumnVector,
    Matrix,
    RowVector,
    SquareMatrix,
    as_square_matrix,
    identity_matrix,
)


class GaussJordanElimination:
    """Invert a matrix with Gauss-Jordan elimination.

    References
    - Gauss-Jordan Elimination
    https://mathworld.wolfram.com/Gauss-JordanElimination.html
    - Inverse of a Matrix using Elementary Row Operations:
    https://www.mathsisfun.com/algebra/matrix-inverse-row-operations-gauss-jordan.html
    - Matrix Inverse
    https://github.com/ThomIves/MatrixInverse
    """

    @classmethod
    def invert(cls, matrix: SquareMatrix) -> SquareMatrix:
        # The left hand-side of augmented matrix. Typically called A.
        left = dataclasses.replace(matrix)  # Copy the matrix
        # The right hand-side of augmented matrix. Typically called I.
        right = identity_matrix(left.n_rows)
        return cls._eliminate(left, right)

    @classmethod
    def _eliminate(
        cls, left: SquareMatrix, right: SquareMatrix, row_index: int = 0
    ) -> SquareMatrix:
        """Apply Gaussian elimination to turn the left matrix to the identity matrix."""
        if row_index == left.n_rows:
            return right

        (left1, right1) = cls._divide_row_with_diagonal(left, right, row_index)
        (left2, right2) = cls._subtract_row_from_other_rows(left1, right1, row_index)

        return cls._eliminate(left2, right2, row_index + 1)

    @classmethod
    def _divide_row_with_diagonal(
        cls, left: SquareMatrix, right: SquareMatrix, row_index: int
    ) -> Tuple[SquareMatrix, SquareMatrix]:
        """Divide the row with its diagonal element.

        So that the diagonal element of row becomes 1.

        Example: When row_index = 0
            [[3, 2, 1],      [[1, 2/3, 1/3],  # Divided by 3
             [2, 1, 3],  =>   [2,   1,   3],  # No change
             [1, 3, 2]]       [1,   3,   2]]  # No change
        """
        left1, right1 = cls.__swap_rows_if_necessary(left, right, row_index)
        scaling_matrix_ = Matrix.from_iterator(
            cls.__scaling_matrix_rows(
                row_index, left.n_rows, 1 / left1.rows[row_index][row_index]
            )
        )
        scaling_matrix = as_square_matrix(scaling_matrix_)

        return (
            as_square_matrix(scaling_matrix * left1),
            as_square_matrix(scaling_matrix * right1),
        )

    @classmethod
    def __swap_rows_if_necessary(
        cls, left: SquareMatrix, right: SquareMatrix, diagonal_index: int
    ) -> Tuple[SquareMatrix, SquareMatrix]:
        """Swap rows so that the `row_index`th diagonal in the left is not zero.

        Example: When diagonal_index = 1
            [[1, 0, 0],      [[1, 0, 0],  # No change
             [0, 0, 1],  =>   [0, 1, 1],  # Swapped with the third row
             [0, 1, 1]]       [0, 0, 1]]  # Swapped with the second row
        """
        potential_targets = tuple(
            i
            for i in range(diagonal_index, left.n_rows)
            if abs(left.rows[i][diagonal_index]) > 1e-8
        )
        if len(potential_targets) == 0:
            raise RuntimeError("Elimination is numerically unstable. Giving up.")

        target = potential_targets[0]
        if target == diagonal_index:
            return left, right

        permutation_matrix_ = Matrix.from_iterator(
            cls.__permutation_matrix_rows(
                (diagonal_index, potential_targets[0]), left.n_rows
            )
        )
        permutation_matrix = as_square_matrix(permutation_matrix_)
        return (
            as_square_matrix(permutation_matrix * left),
            as_square_matrix(permutation_matrix * right),
        )

    @classmethod
    def __permutation_matrix_rows(
        cls, rows_to_swap: Tuple[int, int], n_rows: int
    ) -> Iterator[Iterator[float]]:
        for row_index in range(n_rows):
            if row_index not in rows_to_swap:
                yield cls.__permutation_matrix_row(row_index, n_rows)
            elif row_index == rows_to_swap[0]:
                yield cls.__permutation_matrix_row(rows_to_swap[1], n_rows)
            elif row_index == rows_to_swap[1]:
                yield cls.__permutation_matrix_row(rows_to_swap[0], n_rows)

    @staticmethod
    def __permutation_matrix_row(target_row: int, n_columns: int) -> Iterator[float]:
        for column_index in range(n_columns):
            if column_index == target_row:
                yield 1.0
            else:
                yield 0.0

    @classmethod
    def __scaling_matrix_rows(
        cls, row_index_to_scale: int, n_rows: int, scale: float
    ) -> Iterator[Iterator[float]]:
        for row_index in range(n_rows):
            yield cls.__scaling_matrix_row(row_index, n_rows, row_index_to_scale, scale)

    @staticmethod
    def __scaling_matrix_row(
        row_index: int, n_columns: int, row_index_to_scale, scale: float
    ) -> Iterator[float]:
        for column_index in range(n_columns):
            if row_index != column_index:
                yield 0.0
            elif row_index == row_index_to_scale:
                yield scale
            else:
                yield 1.0

    @staticmethod
    def _subtract_row_from_other_rows(
        left: SquareMatrix, right: SquareMatrix, row_index: int
    ) -> Tuple[SquareMatrix, SquareMatrix]:
        """Subtract a multiple of the row from the other rows.

        So that the `row_index`th column of left matrix have the value of zero
        on the off-diagonal.

        Example: When row_index = 0
            [[1, 2, 3],      [[1,   2,   3],  # No change
             [2, 3, 1],  =>   [0, 3-2, 1-3],  # Subtracted 2 * first row
             [3, 2, 1]]       [0, 2-6, 1-9]]  # Subtracted 3 * first row
        """
        row_multiplier = ColumnVector(
            tuple(
                (
                    0.0
                    if i == row_index
                    else row[row_index] / left.rows[row_index][row_index],
                )
                for i, row in enumerate(left.rows)
            )
        )

        # The 1st row to subtract from the left is row_multiplier[0] * left.row[index]
        # The 2nd row to subtract from the left is row_multiplier[1] * left.row[index]
        to_subtract_from_left = as_square_matrix(
            row_multiplier * RowVector((left.rows[row_index],))
        )
        to_subtract_from_right = as_square_matrix(
            row_multiplier * RowVector((right.rows[row_index],))
        )
        return left - to_subtract_from_left, right - to_subtract_from_right
