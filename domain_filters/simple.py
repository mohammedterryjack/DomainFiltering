from typing import List

from numpy import array, ndarray


class SimpleDomainFilter:
    def __init__(self, min_radius: int = 1, max_radius: int = 15) -> None:
        self._min_radius = min_radius
        self._max_radius = max_radius

    def classify_spacetime(self, spacetime: List[List[int]]) -> List[List[bool]]:
        return list(
            map(lambda lattice: self.classify_lattice(lattice=lattice), spacetime)
        )

    def classify_lattice(self, lattice: List[int]) -> List[bool]:
        return list(
            map(
                lambda index: self.classify_element(index=index, lattice=lattice),
                range(len(lattice)),
            )
        )

    def classify_element(self, lattice: List[int], index: int) -> bool:
        lattice = array(lattice)
        for radius in range(self._min_radius, self._max_radius):
            if self._equal_neighbours(index=index, lattice=lattice, radius=radius):
                return True
        return False

    @staticmethod
    def _equal_neighbours(index: int, lattice: ndarray, radius: int) -> bool:
        min_index, max_index = 0, len(lattice)
        left_neighbours = SimpleDomainFilter._nearest_neighbours(
            start_index=index - radius,
            end_index=index + 1,
            min_index=min_index,
            max_index=max_index,
            lattice=lattice,
        )
        right_neighbours = SimpleDomainFilter._nearest_neighbours(
            start_index=index,
            end_index=index + radius + 1,
            min_index=min_index,
            max_index=max_index,
            lattice=lattice,
        )
        return left_neighbours == right_neighbours

    @staticmethod
    def _nearest_neighbours(
        start_index: int,
        end_index: int,
        min_index: int,
        max_index: int,
        lattice: ndarray,
    ) -> List[int]:
        left_pad = lattice[start_index:].tolist() if start_index < min_index else []
        right_pad = (
            lattice[: end_index - max_index].tolist() if end_index > max_index else []
        )
        return left_pad + lattice[start_index:end_index].tolist() + right_pad
