from ..matrix import SquareMatrix, as_square_matrix, identity_matrix
from .gaussian_elimination import solve


def invert(m: SquareMatrix) -> SquareMatrix:
    """Invert a square-matrix.

    This function solves M * X = I for X.
    """
    inverted, _ = solve(m, identity_matrix(m.n_rows))
    return as_square_matrix(inverted)
