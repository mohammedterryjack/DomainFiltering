from gzip import compress

from numpy import ndarray, zeros_like

approx_kolmogorov_complexity = lambda text: len(compress(text.encode("utf-8")))


def NCD(x: str, y: str) -> float:
    K_x = approx_kolmogorov_complexity(x)
    K_y = approx_kolmogorov_complexity(y)
    K_min = min(K_x, K_y)
    K_max = max(K_x, K_y)
    distance_information = approx_kolmogorov_complexity(x + y) - K_min
    return distance_information / K_max


def vectorise_neighbourhood_1d(spacetime: ndarray, x: int, y: int, r: int) -> list[int]:
    return [spacetime[(y, x_i)] for x_i in range(x - r, x + r + 1)]



def vectorise_neighbourhood(
    spacetime: ndarray,
    x: int,
    y: int,
) -> list[int]:
    """Hilbert Curve for 2D light cone"""
    coordinates_lightcone_hilbert_curve = [
        (y - 2, x + 2),
        (y - 1, x + 1),
        (y, x),
        (y - 1, x - 1),
        (y - 1, x),
        (y - 2, x - 1),
        (y - 2, x),
        (y - 2, x - 2),
    ]
    return [spacetime[coord] for coord in coordinates_lightcone_hilbert_curve]


# def vectorise_neighbourhood_2d(
#     spacetime: ndarray, x: int, y: int,
# ) -> list[int]:
#     """Hilbert Curve for 3x3 grid to map 2d to 1d

#      ___
#      __| |
#     |____|
#     """
#     coordinates_hilbert_curve = [
#         (y - 1, x - 1),
#         (y - 1, x),
#         (y, x),
#         (y, x - 1),
#         (y + 1, x - 1),
#         (y + 1, x),
#         (y + 1, x + 1),
#         (y, x + 1),
#         (y - 1, x + 1),
#     ]
#     vector = []
#     for coordinate in coordinates_hilbert_curve:
#         cell = spacetime[coordinate]
#         vector.append(cell)
#     return vector


def normalised_compression_distance(
    spacetime: list[list[int]],
    x: int,
    y: int,
) -> float:
    return NCD(
        x="".join(
            map(
                str,
                vectorise_neighbourhood(
                    spacetime=spacetime,
                    x=x,
                    y=y-1,
                ),
            )
        ),
        y="".join(
            map(
                str,
                vectorise_neighbourhood(
                    spacetime=spacetime, x=x, y=y
                ),
            )
        ),
    )


def local_ncd(spacetime: ndarray, neighbourhood_radius: int = 1) -> ndarray:
    """Uses past lightcone for input"""
    filtered = zeros_like(spacetime, dtype="float32")
    t, w = filtered.shape
    for y_ in range(neighbourhood_radius + 3, t):
        for x_ in range(neighbourhood_radius + 3, w - 3):
            filtered[y_, x_] = normalised_compression_distance(
                spacetime=spacetime,
                x=x_,
                y=y_,
            )
    return filtered


# from eca import OneDimensionalElementaryCellularAutomata
# from matplotlib.pyplot import show, subplots
# from scipy.stats import mode

# ca = OneDimensionalElementaryCellularAutomata(lattice_width=500)
# for _ in range(300):
#     ca.transition(110)

# spacetime = ca.evolution()
# filtered = local_ncd(spacetime)

# thresh, _ = mode(filtered, axis=None)

# fig, axs = subplots(3)
# axs[0].imshow(spacetime, cmap="gray")
# axs[1].imshow(filtered, cmap="gray")
# axs[2].imshow(filtered < thresh, cmap="gray")
# show()
