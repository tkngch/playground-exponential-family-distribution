from functools import reduce

from ..matrix import SquareMatrix, identity_matrix
from .gaussian_elimination import solve


def determinant(m: SquareMatrix) -> float:
    """Calculate the determinant of matrix.

    This function first derives a series of elementary row operations (E1, E2,
    ...), such that
    ```
    E1 * E2 * ... * En * M
        = I
    ```

    Then both side of equation is multiplied with the inverse of operator
    matrices:
    ```
    En^-1 * ... * E2^-1 * E1^-1 * E1 * E2 * ... En * M
        = En^-1 * ... * E2^-1 * E1^-1 * I
    ```
    which gives us
    ```
    M = En^-1 * ... * E2^-1 * E1^-1
    ```

    Finally, the determinant is given by
    ```
    det(M) = det(En^-1) * ... * det(E2^-1) * det(E1^-1)
    ```
    """
    _, operators = solve(m, identity_matrix(m.n_rows))
    inverted_operator_determinants = map(
        lambda x: x.inverted().matrix_determinant, operators
    )
    return reduce(lambda a, b: a * b, inverted_operator_determinants)
