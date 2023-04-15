from __future__ import annotations

import dataclasses
from abc import ABC
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Tuple, TypeVar, Union

Self = TypeVar("Self", bound="Matrix")
RealNumber = Union[float, int]


@dataclass(frozen=True)
class Matrix(ABC):
    rows: Tuple[Tuple[float, ...], ...]

    @staticmethod
    def make(rows: Tuple[Tuple[float, ...], ...]) -> Matrix:
        n_rows = len(rows)
        n_columns = len(rows[0])
        if n_rows == 1:
            return RowVector(rows)
        if n_columns == 1:
            return ColumnVector(rows)
        if n_rows == n_columns:
            return SquareMatrix(rows)
        return RectangularMatrix(rows)

    @staticmethod
    def from_iterator(rows: Iterator[Iterator[float]]) -> Matrix:
        return Matrix.make(tuple(tuple(row) for row in rows))

    def __post_init__(self):
        n_columns = frozenset([len(row) for row in self.rows])
        if len(n_columns) > 1:
            raise ValueError(f"The rows have inconsistent sizes: Found {n_columns}")

        if self.n_rows == 0 or self.n_columns == 0:
            raise ValueError("An empty matrix is not supported.")

    def _replace_rows(self: Self, rows: Iterator[Iterator[float]]) -> Self:
        new_rows = tuple(tuple(row) for row in rows)

        if self.n_rows != len(new_rows) or self.n_columns != len(new_rows[0]):
            raise ValueError(
                "The replacement rows do not have the same shape as the source matrix. "
                f"Expected {self.n_rows} by {self.n_columns}, but "
                f"got {len(new_rows)} by {len(new_rows[0])}."
            )

        return dataclasses.replace(self, rows=new_rows)

    @property
    def n_rows(self):
        return len(self.rows)

    @property
    def n_columns(self):
        return len(self.rows[0])

    @property
    def columns(self) -> Iterator[Iterator[float]]:
        for column_index in range(self.n_columns):
            yield self._column(column_index)

    def _column(self, column_index: int) -> Iterator[float]:
        for row in self.rows:
            yield row[column_index]

    def transpose(self) -> Matrix:
        return self.from_iterator(self.columns)

    def __round__(self: Self, n: Optional[int] = None) -> Self:
        return self._replace_rows(
            self._apply_function_to_single_elements(lambda x: float(round(x, n)))
        )

    def __add__(self: Self, other: Self) -> Self:
        """Add this vector to the other: `this matrix + other`."""
        if self.n_rows != other.n_rows or self.n_columns != other.n_columns:
            raise ValueError(
                f"Cannot add the {self.n_rows}-by-{self.n_columns} matrix "
                f"to the {other.n_rows}-by-{other.n_columns} matrix."
            )
        return self._replace_rows(
            self._apply_function_to_two_elements(other, lambda this, that: this + that)
        )

    def __radd__(self, other: Union[float, int]) -> Matrix:
        """Add the other to this vector: `other + this matrix`."""
        return self._replace_rows(
            self._apply_function_to_single_elements(lambda x: other + x)
        )

    def __sub__(self: Self, other: Self) -> Self:
        """Subtract the other from this vector to the other: `this matrix - other`."""
        if self.n_rows != other.n_rows or self.n_columns != other.n_columns:
            raise ValueError(
                f"Cannot subtract the {other.n_rows}-by-{other.n_columns} matrix "
                f"from the {self.n_rows}-by-{self.n_columns} matrix."
            )
        return self._replace_rows(
            self._apply_function_to_two_elements(other, lambda this, that: this - that)
        )

    def __rsub__(self, other: Union[float, int]) -> Matrix:
        """Subtract this matrix from the other: `other - this matrix`."""
        return self._replace_rows(
            self._apply_function_to_single_elements(lambda x: other - x)
        )

    def __mul__(self, other: Matrix) -> Matrix:
        """Multiply this matrix with the other: `matrix * other`."""
        if self.n_columns != other.n_rows:
            raise ValueError(
                f"Cannot multiply the {self.n_rows}-by-{self.n_columns} matrix "
                f"with the {other.n_rows}-by-{other.n_columns} matrix."
            )
        return self.from_iterator(self._multiply_with_matrix(other))

    def __rmul__(self: Self, other: Union[float, int]) -> Self:
        """Multiply the other with this matrix: `other * matrix`."""
        return self._replace_rows(
            self._apply_function_to_single_elements(lambda x: other * x)
        )

    def _apply_function_to_single_elements(
        self, f: Callable[[float], float]
    ) -> Iterator[Iterator[float]]:
        """Return `A`, where `A[i][j] = f(self.rows[i][j])`."""
        for row_index in range(self.n_rows):
            yield self.__apply_function_to_single_elements_in_one_row(f, row_index)

    def _apply_function_to_two_elements(
        self: Self, other: Self, f: Callable[[float, float], float]
    ) -> Iterator[Iterator[float]]:
        """Return `A`, where `A[i][j] = f(self.rows[i][j], other.rows[i][j])`."""
        for row_index in range(self.n_rows):
            yield self.__apply_function_to_two_elements_in_one_row(other, f, row_index)

    def __apply_function_to_single_elements_in_one_row(
        self, f: Callable[[float], float], row_index: int
    ) -> Iterator[float]:
        for value in self.rows[row_index]:
            yield f(value)

    def __apply_function_to_two_elements_in_one_row(
        self: Self, other: Self, f: Callable[[float, float], float], row_index: int
    ):
        for this, that in zip(self.rows[row_index], other.rows[row_index]):
            yield f(this, that)

    def _multiply_with_matrix(self, other: Matrix) -> Iterator[Iterator[float]]:
        for row_index in range(self.n_rows):
            yield self.__multiply_row_with_matrix(row_index, other)

    def __multiply_row_with_matrix(
        self, row_index: int, other: Matrix
    ) -> Iterator[float]:
        for column in other.columns:
            dot_product = sum(
                [
                    row_value * column_value
                    for row_value, column_value in zip(self.rows[row_index], column)
                ]
            )
            yield dot_product


@dataclass(frozen=True)
class RowVector(Matrix):
    def __post_init__(self):
        if self.n_rows != 1:
            raise ValueError(
                "A row vector cannot have more than 1 rows: "
                f"found {self.n_rows} rows."
            )
        super().__post_init__()


@dataclass(frozen=True)
class ColumnVector(Matrix):
    def __post_init__(self):
        if self.n_columns != 1:
            raise ValueError(
                "A column vector cannot have more than 1 columns: "
                f"found {self.n_columns} columns."
            )
        super().__post_init__()


@dataclass(frozen=True)
class SquareMatrix(Matrix):
    def __post_init__(self):
        super().__post_init__()
        if self.n_rows != self.n_columns:
            raise ValueError(
                f"A {self.n_rows}-by-{self.n_columns} matrix is not square."
            )


@dataclass(frozen=True)
class RectangularMatrix(Matrix):
    def __post_init__(self):
        super().__post_init__()
        if self.n_rows == self.n_columns:
            raise ValueError(
                f"A {self.n_rows}-by-{self.n_columns} matrix is square "
                "and not rectangular."
            )


def matrix(*rows: Tuple[RealNumber, ...]) -> Matrix:
    return Matrix.make(tuple(tuple(float(val) for val in row) for row in rows))


def square_matrix(*rows: Tuple[Union[float, int], ...]) -> SquareMatrix:
    return as_square_matrix(matrix(*rows))


def column_vector(*elements: RealNumber) -> ColumnVector:
    return ColumnVector(tuple((float(element),) for element in elements))


def identity_matrix(n_rows: int) -> SquareMatrix:
    return as_square_matrix(Matrix.from_iterator(__identity_matrix_rows(n_rows)))


def __identity_matrix_rows(n_rows: int) -> Iterator[Iterator[float]]:
    for row_index in range(n_rows):
        yield __identity_matrix_row(row_index, n_rows)


def __identity_matrix_row(row_index: int, n_columns: int) -> Iterator[float]:
    for column_index in range(n_columns):
        if row_index == column_index:
            yield 1.0
        else:
            yield 0.0


def as_square_matrix(m: Matrix) -> SquareMatrix:
    return SquareMatrix(m.rows)


def as_scalar(m: Matrix) -> float:
    if m.n_rows == 1 and m.n_columns == 1:
        return m.rows[0][0]

    raise ValueError(f"Not a scalar: {m.n_rows}-by-{m.n_columns} matrix.")
