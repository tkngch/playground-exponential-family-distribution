from typing import Iterator, Tuple

from ..matrix import Matrix, SquareMatrix
from .elementary_row_operator import (
    ElementaryAxpy,
    ElementaryMultiplication,
    ElementaryPermutation,
    ElementaryRowOperator,
    apply_operators,
)


def solve(
    a: SquareMatrix, y: Matrix
) -> Tuple[Matrix, Tuple[ElementaryRowOperator, ...]]:
    """Solve A * X = Y for X.

    This function pre-multiplies the square matrix A with a series of
    elementary row operators (E1, E2, ...). For convenience, let
    ```
    E1 * E2 * ... * En = E
    ```

    These operators are constructed such that
    ```
    E * A = I
    ```

    Thus, X is given by
    ```
    E * A * X = E * Y
    <=> X = E * Y
    ```

    The returned values are the solution X along with a series of row operators
    (E1, E2, ...).
    """
    return _operate_on_one_column(a, 0, y, ())


def _operate_on_one_column(
    m: SquareMatrix,
    column_index: int,
    result: Matrix,
    applied_operators: Tuple[ElementaryRowOperator, ...],
) -> Tuple[Matrix, Tuple[ElementaryRowOperator, ...]]:
    """Set the diagonal to one, leaving the other elements at zero in one column.

    To illustrate, suppose column_index = 0 and m is
    ```
    [[0, 1, 2],
     [2, 2, 4],
     [3, 1, 1]]
    ```

    As the (0, 0)th element is zero, a partial pivot is applied, to obtain
    ```
    [[2, 2, 4],
     [0, 1, 2],
     [3, 1, 1]]
    ```

    Then the first row is scaled, such that the (0, 0)th element is one:
    ```
    [[1, 1, 2],
     [0, 1, 2],
     [3, 1, 1]]
    ```

    Then the multiplied first row is subtracted from other rows, such that the
    first column is zero for the second and third rows:
    ```
    [[1,  1,  2],
     [0,  1,  2],
     [0, -2, -5]]
    ```

    The same operations are applied to the result matrix.
    """
    if m.n_columns == column_index:
        return result, applied_operators

    pivot_operator = __partial_pivot_operator(m, column_index)
    pivoted = pivot_operator.apply(m)

    operators = (__diagonal_scaling_operator(pivoted, column_index),) + tuple(
        __non_diagonal_zeroing_operators(pivoted, column_index)
    )

    return _operate_on_one_column(
        apply_operators(pivoted, operators),
        column_index + 1,
        apply_operators(pivot_operator.apply(result), operators),
        applied_operators + (pivot_operator,) + operators,
    )


def __partial_pivot_operator(m: SquareMatrix, index: int) -> ElementaryPermutation:
    if abs(m.rows[index][index]) > 1e-4:
        return ElementaryPermutation.noop()

    for row_index in range(index, m.n_rows):
        if abs(m.rows[row_index][index]) > 1e-4:
            return ElementaryPermutation(
                indices_of_rows_to_interchange=(index, row_index),
            )

    raise RuntimeError(
        f"Cannot find an element that is large enough for column {index}"
    )


def __non_diagonal_zeroing_operators(
    m: SquareMatrix, index: int
) -> Iterator[ElementaryAxpy]:
    for row_index in range(m.n_rows):
        if row_index == index:
            continue

        yield ElementaryAxpy(
            scalar=-1 * m.rows[row_index][index],
            index_of_row_to_multiply=index,
            index_of_row_to_replace=row_index,
        )


def __diagonal_scaling_operator(
    m: SquareMatrix, index: int
) -> ElementaryMultiplication:
    return ElementaryMultiplication(
        index_of_row_to_multiply=index,
        scalar=1.0 / m.rows[index][index],
    )
