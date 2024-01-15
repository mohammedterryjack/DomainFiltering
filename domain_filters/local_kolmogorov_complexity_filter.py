from gzip import compress

from numpy import ndarray, zeros_like


def vectorise_neighbourhood_1d(
    spacetime: ndarray, x: int, y: int, r:int
) -> list[int]:
    return [
        spacetime[(y,x_i)] for x_i in range(x-r,x+r+1)
    ]


def vectorise_neighbourhood_2d(
    spacetime: ndarray, x: int, y: int,
) -> list[int]:
    """Hilbert Curve for 3x3 grid to map 2d to 1d

     ___
     __| |
    |____|
    """
    coordinates_hilbert_curve = [
        (y - 1, x - 1),
        (y - 1, x),
        (y, x),
        (y, x - 1),
        (y + 1, x - 1),
        (y + 1, x),
        (y + 1, x + 1),
        (y, x + 1),
        (y - 1, x + 1),
    ]
    vector = []
    for coordinate in coordinates_hilbert_curve:
        cell = spacetime[coordinate]
        vector.append(cell)
    return vector

def normalised_compression_distance(spacetime: list[list[int]], x: int, y: int, neighbourhood_radius:int) -> float:
    approx_kolmogorov_complexity = lambda text: len(compress(text.encode('utf-8')))
    def NCD(x:str, y:str) -> float:
        K_x = approx_kolmogorov_complexity(x)
        K_y = approx_kolmogorov_complexity(y)
        K_min = min(K_x, K_y)
        K_max = max(K_x, K_y)
        distance_information = approx_kolmogorov_complexity(x+y) - K_min
        return distance_information/K_max
    return NCD(
        x="".join(map(str, vectorise_neighbourhood_1d(
            spacetime=spacetime, x=x, y=y, r=neighbourhood_radius
        ))),
        y="".join(map(str, vectorise_neighbourhood_1d(
            spacetime=spacetime, x=x, y=y-1, r=neighbourhood_radius
        )))
    )

def complexity(spacetime: list[list[int]], x: int, y: int) -> float:
    vector = vectorise_neighbourhood_2d(
        spacetime=spacetime, x=x, y=y
    )
    vector_str = "".join(map(str, vector))
    vector_bytes = vector_str.encode("utf-8")
    total = len(vector_bytes)
    header = len(compress("".encode("utf-8")))
    n = len(compress(vector_bytes))
    return (n - header) / total


def local_ncd(
    spacetime: ndarray, neighbourhood_radius: int = 1
) -> ndarray:
    filtered = zeros_like(spacetime, dtype="float32")
    t, w = filtered.shape
    for y_ in range(neighbourhood_radius, t - neighbourhood_radius):
        for x_ in range(neighbourhood_radius, w - neighbourhood_radius):
            filtered[y_, x_] = normalised_compression_distance(
                spacetime=spacetime, x=x_, y=y_, 
                neighbourhood_radius=neighbourhood_radius
            )
    return filtered

def local_kolmogorov_complexity(
    spacetime: ndarray, neighbourhood_radius:int=2
) -> ndarray:
    filtered = zeros_like(spacetime, dtype="float32")
    t, w = filtered.shape
    for y_ in range(neighbourhood_radius, t - neighbourhood_radius):
        for x_ in range(neighbourhood_radius, w - neighbourhood_radius):
            filtered[y_, x_] = complexity(
                spacetime=spacetime, x=x_, y=y_, 
            )
    return filtered


from eca import OneDimensionalElementaryCellularAutomata
from matplotlib.pyplot import subplots, show

ca = OneDimensionalElementaryCellularAutomata(lattice_width=1000)
for _ in range(300):
    ca.transition(110)

spacetime = ca.evolution()
filtered = local_kolmogorov_complexity(spacetime)
filtered2 = local_ncd(spacetime, neighbourhood_radius=1)

fig, axs = subplots(3)
axs[0].imshow(spacetime, cmap="gray")
axs[1].imshow(filtered,cmap="gray")
axs[2].imshow(filtered2, cmap="gray")
show()
