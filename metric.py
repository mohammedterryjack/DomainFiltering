from hilbert import decode
from numpy import ndarray, pad
from scipy.spatial.distance import cosine


def hilbert_flatten(matrix: ndarray, n_iterations: int = 8) -> ndarray:
    """flatten 2d matrix into 1d vector using Hilbert Curve"""
    h, w = matrix.shape
    length = max(h, w)
    y_pad, x_pad = length - h, length - w
    matrix_padded = pad(
        matrix, ((0, y_pad), (0, x_pad)), mode="constant", constant_values=0
    )
    matrix_dimensions = 2
    coordinates = tuple(
        decode(range(h * w), matrix_dimensions, n_iterations).T.tolist()
    )
    return matrix_padded[coordinates]


def get_score(predicted: ndarray, expected: ndarray) -> float:
    predicted_vector = hilbert_flatten(matrix=predicted)
    expected_vector = hilbert_flatten(matrix=expected)

    distance = cosine(predicted_vector, expected_vector)
    return 1 - distance
