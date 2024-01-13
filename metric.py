from hilbert import decode
from numpy import ndarray
from scipy.spatial.distance import cosine


def hilbert_flatten(matrix: ndarray, n_iterations: int = 8) -> ndarray:
    """flatten 2d matrix into 1d vector using Hilbert Curve"""
    h, w = matrix.shape
    matrix_dimensions = 2
    coordinates = tuple(
        decode(range(h * w), matrix_dimensions, n_iterations).T.tolist()
    )
    return matrix[coordinates]


def get_score(predicted: ndarray, expected: ndarray) -> float:
    predicted_vector = hilbert_flatten(matrix=predicted)
    expected_vector = hilbert_flatten(matrix=expected)

    score = cosine(predicted_vector, expected_vector)
    # TODO: change this to cross-entropy-loss
    # TODO: there is something buggy here - ensure it doesnt go outside valid coordinates
    return score
