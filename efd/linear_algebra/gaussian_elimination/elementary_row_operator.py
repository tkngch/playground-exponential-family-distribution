from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Iterator, Tuple, TypeVar, cast

from ..matrix import Matrix, SquareMatrix, as_square_matrix

T = TypeVar("T", bound=Matrix)
Self = TypeVar("Self", bound="ElementaryRowOperator")


class ElementaryRowOperator(ABC):
    def apply(self, m: T) -> T:
        """Apply the operator to the matrix."""
        return cast(T, self._matrix(m.n_rows) * m)

    @abstractmethod
    def inverted(self: Self) -> Self:
        """Make an inverse operator."""

    @abstractmethod
    def _matrix(self, n_rows: int) -> SquareMatrix:
        """Derive the operator matrix with the specified number of rows.

        To apply the elementary row operation on X, pre-multiply X by the
        operator matrix.
        """

    @property
    @abstractmethod
    def matrix_determinant(self) -> float:
        """Calculate the determinant of operator matrix."""


@dataclass(frozen=True)
class ElementaryPermutation(ElementaryRowOperator):
    """Interchange two rows.

    To illustrate, a matrix to swap the first row and the second row is given by
    ```
    [[0, 1, 0],
     [1, 0, 1],
     [0, 0, 1]]
    ```
    """

    indices_of_rows_to_interchange: Tuple[int, int]

    @staticmethod
    def noop() -> ElementaryPermutation:
        return ElementaryPermutation((0, 0))

    def inverted(self: Self) -> Self:
        return self

    def _matrix(self, n_rows: int) -> SquareMatrix:
        return as_square_matrix(Matrix.from_iterator(self.__matrix_rows(n_rows)))

    @property
    def matrix_determinant(self) -> float:
        if (
            self.indices_of_rows_to_interchange[0]
            == self.indices_of_rows_to_interchange[1]
        ):
            return 1.0
        return -1.0

    def __matrix_rows(self, n_rows: int) -> Iterator[Iterator[float]]:
        for row_index in range(n_rows):
            if row_index not in self.indices_of_rows_to_interchange:
                yield self.__matrix_row(row_index, n_rows)
            elif row_index == self.indices_of_rows_to_interchange[0]:
                yield self.__matrix_row(self.indices_of_rows_to_interchange[1], n_rows)
            elif row_index == self.indices_of_rows_to_interchange[1]:
                yield self.__matrix_row(self.indices_of_rows_to_interchange[0], n_rows)

    @staticmethod
    def __matrix_row(row_index: int, n_columns: int) -> Iterator[float]:
        for column_index in range(n_columns):
            if column_index == row_index:
                yield 1.0
            else:
                yield 0.0


@dataclass(frozen=True)
class ElementaryMultiplication(ElementaryRowOperator):
    """Multiply one row with a scalar.

    For example, a matrix to multiply the first row with 0.5 is given by
    ```
    [[0.5, 0, 0],
     [ 0,  1, 0],
     [ 0,  0, 1]]
    ```
    """

    index_of_row_to_multiply: int
    scalar: float

    def inverted(self: ElementaryMultiplication) -> ElementaryMultiplication:
        return replace(self, scalar=1.0 / self.scalar)

    def _matrix(self, n_rows: int) -> SquareMatrix:
        return as_square_matrix(Matrix.from_iterator(self.__matrix_rows(n_rows)))

    @property
    def matrix_determinant(self) -> float:
        return self.scalar

    def __matrix_rows(self, n_rows: int) -> Iterator[Iterator[float]]:
        for row_index in range(n_rows):
            yield self.__matrix_row(row_index, n_rows)

    def __matrix_row(self, row_index: int, n_columns: int) -> Iterator[float]:
        for column_index in range(n_columns):
            if row_index != column_index:
                yield 0.0
            elif row_index == self.index_of_row_to_multiply:
                yield self.scalar
            else:
                yield 1.0


@dataclass(frozen=True)
class ElementaryAxpy(ElementaryRowOperator):
    """Add a multiple of one row to another.

    Let `a`, `x`, and `y` denote a scalar, a row-vector, and a row-vector. Then
    this operator replaces y with ax + y.

    For example, a matrix to add -2 * the first row to the second row is given by
    ```
    [[1,  0, 0],
     [-2, 1, 0],
     [0,  0, 1]]
    ```

    This is a Frobenius matrix, whose inverse equals to the original matrix
    with changed signs outside the main diagonal:
    ```
    [[1, 0, 0],
     [2, 1, 0],
     [0, 0, 1]]
    ```
    """

    scalar: float
    index_of_row_to_multiply: int
    index_of_row_to_replace: int

    def inverted(self: ElementaryAxpy) -> ElementaryAxpy:
        return replace(self, scalar=-1 * self.scalar)

    def _matrix(self, n_rows: int) -> SquareMatrix:
        return as_square_matrix(Matrix.from_iterator(self.__matrix_rows(n_rows)))

    @property
    def matrix_determinant(self) -> float:
        return 1.0

    def __matrix_rows(self, n_rows: int) -> Iterator[Iterator[float]]:
        for row_index in range(n_rows):
            yield self.__matrix_row(row_index, n_rows)

    def __matrix_row(self, row_index: int, n_columns) -> Iterator[float]:
        for column_index in range(n_columns):
            if row_index == column_index:
                yield 1.0

            elif (
                row_index == self.index_of_row_to_replace
                and column_index == self.index_of_row_to_multiply
            ):
                yield self.scalar

            else:
                yield 0.0


def apply_operators(m: T, operators: Tuple[ElementaryRowOperator, ...]) -> T:
    if len(operators) == 0:
        return m
    return apply_operators(operators[0].apply(m), operators[1:])
