from math import isnan

from numpy import ndarray
from ot import dist, emd


def get_score(predicted: ndarray, expected: ndarray) -> float:
    M = dist(predicted, expected)
    earth_movers_distance_2d = emd(predicted, expected, M)
    normalisation_factor = earth_movers_distance_2d.max()
    earth_movers_distance_2d_normalised = (
        earth_movers_distance_2d / normalisation_factor
    )
    distance = (
        earth_movers_distance_2d_normalised.sum()
        / earth_movers_distance_2d_normalised.size
    )
    return 1.0 if isnan(distance) else 1 - distance


# F1 score for 2d Matrix

# loss calculations if you flatten the 2d matrix
# (Similar to language modelling loss) - cross-entropy loss (Good for imbalanced classes), the perplexity loss, and the BLEU score.
