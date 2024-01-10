from collections import defaultdict
from math import log
from typing import Generator

from numpy import mean, ndarray, zeros_like
from scipy.spatial.distance import cosine


def coordinates_lightcone(
    x: int,
    t: int,
    max_space: int,
    max_time: int,
    lightcone_depth: int,
    spread_rate: int = 1,
    allow_partial_cones: bool = True,
) -> Generator[tuple[int, int], None, None]:
    t_end = t + lightcone_depth
    if lightcone_depth < 0:
        if t_end < 0 and not allow_partial_cones:
            return
        rows = range(t - 1, max(t_end, 0), -1)
    else:
        if t_end >= max_time and not allow_partial_cones:
            return
        rows = range(t + 1, min(t_end, max_time))

    for row_index, t_ in enumerate(rows, start=1):
        span = row_index * spread_rate
        leftmost_index, rightmost_index = x - span, x + span

        left_overflow, right_overflow = [], []
        if leftmost_index < 0:
            left_overflow = list(range(max_space + leftmost_index, max_space))
            leftmost_index = 0
        if rightmost_index > max_space:
            right_overflow = list(range(0, rightmost_index - max_space))
            rightmost_index = max_space

        for x_ in (
            left_overflow
            + list(range(leftmost_index, rightmost_index))
            + right_overflow
        ):
            yield (t_, x_)


def cell_values(
    spacetime: ndarray, coordinates: list[tuple[int, int]]
) -> Generator[int, None, None]:
    for coord in coordinates:
        yield spacetime[coord]


def past_lightcones(
    spacetimes: list[ndarray],
    lightcone_depth: int,
) -> dict[str, set[tuple]]:
    past_lightcone_to_future_lightcones = defaultdict(set)
    for spacetime in spacetimes:
        max_time, max_space = spacetime.shape
        for t in range(max_time):
            for x in range(max_space):
                past_lightcone_repr = "".join(
                    map(
                        str,
                        cell_values(
                            spacetime=spacetime,
                            coordinates=coordinates_lightcone(
                                x=x,
                                t=t,
                                max_space=max_space,
                                max_time=max_time,
                                lightcone_depth=-lightcone_depth,
                                allow_partial_cones=True,
                            ),
                        ),
                    )
                )
                future_lightcone = tuple(
                    cell_values(
                        spacetime=spacetime,
                        coordinates=coordinates_lightcone(
                            x=x,
                            t=t,
                            max_space=max_space,
                            max_time=max_time,
                            lightcone_depth=lightcone_depth,
                            allow_partial_cones=False,
                        ),
                    )
                )
                if any(future_lightcone):
                    past_lightcone_to_future_lightcones[past_lightcone_repr].add(
                        future_lightcone
                    )
    return past_lightcone_to_future_lightcones


def average(lightcones: set[tuple]) -> ndarray:
    return mean(list(lightcones), axis=0)


def distance(candidate_lightcone: ndarray, lightcone_distribution: ndarray) -> float:
    return cosine(candidate_lightcone, lightcone_distribution)


def statistical_complexities(
    causal_state_to_past_lightcones: dict[int, list[tuple]],
    past_lightcone_to_causal_state: dict[str, int],
) -> dict[int, float]:
    n_past_lightcones = len(past_lightcone_to_causal_state)
    statistical_comlpexities = {}
    for causal_state, past_lightcones in causal_state_to_past_lightcones.items():
        probability = len(past_lightcones) / n_past_lightcones
        statistical_comlpexities[causal_state] = -log(probability)
    return statistical_comlpexities


def causal_states(
    past_lightcones: dict[str, set[tuple]],
    similarity_threshold: float,
) -> tuple[dict[str, int], dict[int, float]]:
    causal_state_to_future_lightcones = {}
    causal_state_to_past_lightcones = {}
    past_lightcone_to_causal_state = {}

    for past_lightcone_repr, candidate_future_lightcones in past_lightcones.items():
        for (
            causal_state,
            future_lightcones,
        ) in causal_state_to_future_lightcones.items():
            if (
                distance(
                    candidate_lightcone=average(candidate_future_lightcones),
                    lightcone_distribution=average(future_lightcones),
                )
                <= similarity_threshold
            ):
                future_lightcones |= candidate_future_lightcones
                causal_state_to_past_lightcones[causal_state].append(
                    past_lightcone_repr
                )
                past_lightcone_to_causal_state[past_lightcone_repr] = causal_state
        if past_lightcone_repr not in past_lightcone_to_causal_state:
            causal_state = hash(past_lightcone_repr)
            causal_state_to_future_lightcones[
                causal_state
            ] = candidate_future_lightcones
            causal_state_to_past_lightcones[causal_state] = [past_lightcone_repr]
            past_lightcone_to_causal_state[past_lightcone_repr] = causal_state

    causal_state_to_statistical_complexity = statistical_complexities(
        causal_state_to_past_lightcones=causal_state_to_past_lightcones,
        past_lightcone_to_causal_state=past_lightcone_to_causal_state,
    )
    return past_lightcone_to_causal_state, causal_state_to_statistical_complexity


def statistical_complexity(
    spacetimes: list[ndarray],
    lightcone_depth: int,
    causal_state_clustering_similarity_threshold: float,
) -> dict[str, float]:
    past_lightcone_to_future_lightcones = past_lightcones(
        spacetimes=spacetimes, lightcone_depth=lightcone_depth
    )
    (
        past_lightcone_to_causal_state,
        causal_state_to_statistical_complexity,
    ) = causal_states(
        past_lightcones=past_lightcone_to_future_lightcones,
        similarity_threshold=causal_state_clustering_similarity_threshold,
    )
    past_lightcone_to_statistical_complexity = {}
    for past_lightcone_repr, causal_state in past_lightcone_to_causal_state.items():
        past_lightcone_to_statistical_complexity[
            past_lightcone_repr
        ] = causal_state_to_statistical_complexity[causal_state]
    return past_lightcone_to_statistical_complexity


def local_statistical_complexity_filter(
    spacetime: ndarray,
    past_lightcone_to_statistical_complexity: dict[str, float],
    lightcone_depth: int,
) -> ndarray:
    filtered_spacetime = zeros_like(spacetime, dtype="float64")
    max_time, max_space = spacetime.shape
    for t in range(max_time):
        for x in range(max_space):
            past_lightcone_repr = "".join(
                map(
                    str,
                    cell_values(
                        spacetime=spacetime,
                        coordinates=coordinates_lightcone(
                            x=x,
                            t=t,
                            max_space=max_space,
                            max_time=max_time,
                            lightcone_depth=-lightcone_depth,
                            allow_partial_cones=True,
                        ),
                    ),
                )
            )
            filtered_spacetime[t, x] = past_lightcone_to_statistical_complexity.get(
                past_lightcone_repr, float("-inf")
            )
    return filtered_spacetime


from cv2 import THRESH_BINARY, THRESH_OTSU, cvtColor, threshold
from eca import OneDimensionalElementaryCellularAutomata
from matplotlib.pyplot import imshow, show


def spacetime(lattice_width: int, time: int, rule_number: int) -> ndarray:
    ca = OneDimensionalElementaryCellularAutomata(lattice_width=lattice_width)
    for _ in range(time):
        ca.transition(rule_number)
    return ca.evolution()


n_spacetimes = 1
lightcone_depth = 5
similarity_theta = 0.05
lattice_width = 100
time = 100
rule_number = 110
# sts = [spacetime(lattice_width=lattice_width,time=time, rule_number=rule_number) for _ in range(n_spacetimes)]
st = spacetime(lattice_width=lattice_width, time=time, rule_number=rule_number)
sts = [st]
past_lightcone_to_statistical_complexity = statistical_complexity(
    spacetimes=sts,
    lightcone_depth=lightcone_depth,
    causal_state_clustering_similarity_threshold=similarity_theta,
)
filtered_spacetime = local_statistical_complexity_filter(
    spacetime=st,
    past_lightcone_to_statistical_complexity=past_lightcone_to_statistical_complexity,
    lightcone_depth=lightcone_depth,
)
imshow(filtered_spacetime)
show()

# TODO: visualise the location of the same past light cone with two different future light cones
