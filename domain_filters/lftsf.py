from typing import List

from numpy import ndarray, zeros
from numpy.fft import irfft2, rfft2
from numpy.lib.stride_tricks import sliding_window_view


class LocalisedFourierTransformSelfFilter:
    def __init__(
        self, localisation_size: int = 4, binarisation_threshold: float = 0.5
    ) -> None:
        self._localisation_size = localisation_size
        self._binarisation_threshold = binarisation_threshold

    def classify_spacetime(self, spacetime: List[List[int]]) -> ndarray:
        filtered_spacetime = zeros(spacetime.shape)
        for i, row in enumerate(
            self._submatrices(
                matrix=spacetime,
                submatrix_width=self._localisation_size,
                submatrix_height=self._localisation_size,
            )
        ):
            for j, submatrix in enumerate(row):
                filtered_spacetime[i][j] = self.classify_submatrix(
                    submatrix=submatrix,
                )
        return filtered_spacetime

    def classify_submatrix(self, submatrix: ndarray) -> bool:
        if not submatrix.any():
            return True
        return self._fourier_transform_self_filter(
            matrix=submatrix, theta=self._binarisation_threshold
        )

    @staticmethod
    def _fourier_transform_self_filter(matrix: ndarray, theta: float) -> bool:
        transformed_matrix = rfft2(matrix)
        self_filter = transformed_matrix**2
        regular_patterns = irfft2(self_filter)
        regular_patterns_normalised = regular_patterns / regular_patterns.max()
        binary_regular_patterns = regular_patterns_normalised > theta
        return binary_regular_patterns[0, 0]

    @staticmethod
    def _submatrices(
        matrix: List[List[int]], submatrix_width: int, submatrix_height: int
    ) -> ndarray:
        return sliding_window_view(
            matrix, window_shape=(submatrix_width, submatrix_height)
        )
