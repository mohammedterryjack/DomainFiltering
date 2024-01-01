from argparse import ArgumentParser
from base64 import b64decode
from json import load

from matplotlib.pyplot import show, subplots
from numpy import array, frombuffer, ndarray, ones_like, where

from domain_filters.contours_via_circles import detect_contours
from domain_filters.lftsf import LocalisedFourierTransformSelfFilter
from domain_filters.simple import SimpleDomainFilter
from emd import get_score
from frequency_filter import filter_by_lookup_frequency


def generate_domain_pattern_from_pattern_signature(
    width: int,
    depth: int,
    pattern_signature: list[str],
) -> list[list[int]]:
    rows = []
    for _ in range(depth):
        for pattern in pattern_signature:
            rows.append(list(map(int, (pattern * width)[:width])))
    return rows[:depth]


def generate_selected_domain_patterns(
    width: int, depth: int, pattern_signatures: list[list[str]]
) -> list[list[int]]:
    domain_patterns = []
    for pattern_signature in pattern_signatures:
        domain_patterns.append(
            generate_domain_pattern_from_pattern_signature(
                width=width,
                depth=depth,
                pattern_signature=pattern_signature,
            )
        )
    return domain_patterns


def fill_domains(
    n_domains: int, segmented_image: ndarray, background_patterns: list[list[list[int]]]
) -> ndarray:
    filled_image = ones_like(segmented_image) * -1
    for domain_label in range(n_domains):
        for x, y in zip(*where(segmented_image == domain_label)):
            background_pattern = background_patterns[domain_label]
            filled_image[x][y] = background_pattern[x][y]
    return filled_image


def string_to_array(image: str, shape: tuple[int, int]) -> ndarray:
    image_bytes = b64decode(image.encode("utf-8"))
    image_array = frombuffer(image_bytes, dtype=int)
    return image_array.reshape(shape).astype(bool).astype(int)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--fourier_localisation_size", type=int, default=4)
    parser.add_argument("--circles_neighbourhood_radius", type=int, default=4)
    parser.add_argument("--circles_threshold", type=float, default=0.2)
    parser.add_argument("--fourier_threshold", type=float, default=0.5)

    arguments = parser.parse_args()

    with open(arguments.path) as json_file:
        data = load(json_file)
    domain_pattern_signatures = [
        domain["pattern_signature"].split("-") for domain in data["metadata"]["domains"]
    ]
    defects = string_to_array(
        image=data["annotated_defects"],
        shape=(data["metadata"]["time"], data["metadata"]["lattice_width"]),
    )
    defects = 1 - defects
    domains = array(data["domain_regions"], dtype=int)
    spacetime = fill_domains(
        n_domains=len(domain_pattern_signatures),
        segmented_image=domains,
        background_patterns=generate_selected_domain_patterns(
            width=data["metadata"]["lattice_width"],
            depth=data["metadata"]["time"],
            pattern_signatures=domain_pattern_signatures,
        ),
    )

    simple_domain_filter = SimpleDomainFilter()
    lftsf_domain_filter = LocalisedFourierTransformSelfFilter(
        localisation_size=arguments.fourier_localisation_size,
        binarisation_threshold=arguments.fourier_threshold,
    )
    prediction_fourier = lftsf_domain_filter.classify_spacetime(spacetime=spacetime)
    prediction_circles = detect_contours(
        image=spacetime,
        neighbourhood_radius=arguments.circles_neighbourhood_radius,
        threshold=arguments.circles_threshold,
    )
    prediction_simple = simple_domain_filter.classify_spacetime(spacetime=spacetime)
    prediction_frequency = filter_by_lookup_frequency(
        spacetime_evolution=spacetime, display=True
    )

    score_fourier = get_score(predicted=prediction_fourier, expected=defects)
    score_circles = get_score(predicted=prediction_circles, expected=defects)
    score_simple = get_score(predicted=array(prediction_simple), expected=defects)
    score_frequency = get_score(predicted=prediction_frequency, expected=defects)

    print(
        f"Scores:\n\tFourier={score_fourier}\n\tCircles={score_circles}\n\tSimple={score_simple}\n\tLookup Frequency = {score_frequency}"
    )

    fig, axs = subplots(6)
    fig.suptitle(arguments.path)
    axs[0].imshow(spacetime, cmap="gray")
    axs[1].imshow(defects, cmap="gray")
    axs[2].imshow(prediction_fourier, cmap="gray")
    axs[3].imshow(prediction_circles, cmap="gray")
    axs[4].imshow(prediction_simple, cmap="gray")
    axs[5].imshow(prediction_frequency, cmap="gray")
    show()
