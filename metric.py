from math import isnan

from numpy import ndarray
from ot import dist

# TODO: flatten 2d matrix into 1d vector
# find cross-entropy loss (Good for imbalanced classes)


def get_score(predicted: ndarray, expected: ndarray) -> float:
    earth_movers_distance_2d = dist(predicted, expected)
    normalisation_factor = earth_movers_distance_2d.max()
    earth_movers_distance_2d_normalised = (
        earth_movers_distance_2d / normalisation_factor
    )
    distance = (
        earth_movers_distance_2d_normalised.sum()
        / earth_movers_distance_2d_normalised.size
    )
    return 1.0 if isnan(distance) else 1 - distance
