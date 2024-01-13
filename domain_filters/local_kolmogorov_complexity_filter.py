from collections import Counter
from gzip import compress

from numpy import ndarray, zeros_like


def vectorise_neighbourhood(
    spacetime: ndarray, x: int, y: int, neighbourhood_radius: int
) -> list[int]:
    """Hilbert Curve for 3x3 grid to map 2d to 1d
    
     ___  
     __| |
    |____|
    """
    #TODO: use it inside a past lightcone instead of a 3x3 window
    coordinates_hilbert_curve = [
        (y-1,x-1), (y-1,x), 
        (y,x), (y,x-1),
        (y+1,x-1), (y+1,x), 
        (y+1,x+1),(y,x+1),(y-1,x+1)
    ]
    vector = []
    for coordinate in coordinates_hilbert_curve:
        cell = spacetime[coordinate]
        vector.append(cell)
    return vector


def complexity(spacetime: list[list[int]], x: int, y: int, r: int) -> float:
    vector = vectorise_neighbourhood(
        spacetime=spacetime, x=x, y=y, neighbourhood_radius=r
    )
    vector_str = "".join(map(str, vector))
    vector_bytes = vector_str.encode("utf-8")
    total = len(vector_bytes)
    header = len(compress("".encode("utf-8")))
    n = len(compress(vector_bytes))
    return (n - header) / total


def local_kolmogorov_complexity(
    spacetime: ndarray, neighbourhood_radius: int = 2
) -> ndarray:
    filtered = zeros_like(spacetime, dtype="float32")
    t, w = filtered.shape
    for y_ in range(neighbourhood_radius, t - neighbourhood_radius):
        for x_ in range(neighbourhood_radius, w - neighbourhood_radius):
            filtered[y_, x_] = complexity(
                spacetime=spacetime, x=x_, y=y_, r=neighbourhood_radius
            )
    return filtered


from eca import OneDimensionalElementaryCellularAutomata
from matplotlib.pyplot import imshow, show

ca = OneDimensionalElementaryCellularAutomata(lattice_width=100)
for _ in range(100):
    ca.transition(110)

spacetime = ca.evolution()

imshow(spacetime)
show()
filtered = local_kolmogorov_complexity(spacetime)
imshow(filtered)
show()