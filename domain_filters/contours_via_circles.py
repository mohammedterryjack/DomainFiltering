from math import pi

from numpy import arctan2, mean, ndarray, zeros_like


def pairwise_combinations(r: int) -> list[tuple[int, int]]:
    """Efficiently generate all possible pairs of integers whose sum is r"""
    combinations = []
    for a in range(r + 1):
        b = r - abs(a)
        combinations.append((a, b))
        if a:
            combinations.append((-a, b))
        if b:
            combinations.append((a, -b))
        if a and b:
            combinations.append((-a, -b))
    return combinations


def circular_neighbourhood(
    centre: tuple[int, int], radius: int, max_width: int, max_height: int
) -> list[tuple[int, int]]:
    """Get neighbouring coordinates that form a circle up to the given radius around the centre coordinate given"""
    x, y = centre
    neighbours = []
    for r in range(radius + 1):
        for dX, dY in pairwise_combinations(r=r):
            x_new = x + dX
            y_new = y + dY
            if 0 <= x_new < max_width and 0 <= y_new < max_height:
                neighbours.append((x_new, y_new))
    return neighbours


def label_coordinates_by_segment_number(
    coordinates: list[tuple[int, int]], n_segments: int = 8
) -> list[int]:
    """Divide the circle into n segments and label each coordinate within the circle with its segment number"""
    X, Y = zip(*coordinates)
    angles = arctan2(Y, X)
    positive_angles = pi + angles
    scale_factor = n_segments / (2 * pi)
    results = positive_angles * scale_factor
    return list(map(int, results))


def distance_between_coordinates(
    coordinates_half1: list[tuple[int, int]],
    coordinates_half2: list[tuple[int, int]],
    image: ndarray,
) -> float:
    """The absolute difference between the means of each set of coordinates"""
    l1_distance = lambda X, Y: abs(mean(X) - mean(Y))
    values1 = [image[y][x] for x, y in coordinates_half1]
    values2 = [image[y][x] for x, y in coordinates_half2]
    if len(values1) > 0 and len(values2) > 0:
        return l1_distance(values1, values2)
    return 0.0


def gradients_between_circle_halves(
    centre_coordinate: tuple[int, int],
    image: ndarray,
    coordinates: list[tuple[int, int]],
) -> list[float]:
    """For each circle with a given orientation, measure the distance between the values of the coordinates in its two halves"""
    x_centre, y_centre = centre_coordinate
    coordinates_offset = [(x - x_centre, y - y_centre) for x, y in coordinates]
    coordinate_labels = label_coordinates_by_segment_number(
        coordinates=coordinates_offset
    )
    return [
        distance_between_coordinates(
            coordinates_half1=[
                coordinate
                for label, coordinate in zip(coordinate_labels, coordinates)
                if label in selected_labels
            ],
            coordinates_half2=[
                coordinate
                for label, coordinate in zip(coordinate_labels, coordinates)
                if label not in selected_labels
            ],
            image=image,
        )
        for selected_labels in {
            "orientation of split: horizontal": [0, 1, 2, 3],
            "orientation of split: diagonal left": [1, 2, 3, 4],
            "orientation of split: vertical": [2, 3, 4, 5],
            "orientation of split: diagonal right": [3, 4, 5, 6],
        }.values()
    ]


def is_contour(
    coordinate: tuple[int, int],
    radius: int,
    image: ndarray,
    difference_threshold: float,
) -> bool:
    """If the maximum difference between two circle halves around the given coordinate is greater than the threshold, assume it sits on a contour"""
    height, width = image.shape
    x_centre, y_centre = coordinate
    coordinates = circular_neighbourhood(
        centre=(x_centre, y_centre), radius=radius, max_width=width, max_height=height
    )
    difference = max(
        gradients_between_circle_halves(
            image=image, centre_coordinate=coordinate, coordinates=coordinates
        )
    )
    return difference < difference_threshold


def detect_contours(
    image: ndarray, neighbourhood_radius: int, threshold: float
) -> ndarray:
    """Draw the contours for the given image using semicircle-difference heuristic"""
    contours = zeros_like(image)
    width, height = contours.shape
    for y in range(width):
        for x in range(height):
            contours[y][x] = is_contour(
                coordinate=(x, y),
                radius=neighbourhood_radius,
                image=image,
                difference_threshold=threshold,
            )
    return contours
