from gzip import compress 
from numpy import zeros_like, ndarray
from collections import Counter

def vectorise_neighbourhood(spacetime:ndarray, x:int, y:int, neighbourhood_radius:int) -> list[int]:
    coordinates = [
        (y,x),
        (y,x-1),(y,x+1),
        (y-1,x),(y+1,x),
        (y-1,x-1),(y+1,x+1),
        (y-1,x+1),(y+1,x-1),
        (y-1,x+1),(y-1,x),(y-1,x-1),(y,x-1),(y+1,x-1),(y+1,x),(y+1,x+1),
    ]
    vector = []
    for coordinate in coordinates:
        cell = spacetime[coordinate]
        vector.append(cell)
    # vector = []
    # for y_ in range(y-neighbourhood_radius,y+neighbourhood_radius+1):
    #   for x_ in range(x-neighbourhood_radius,x+neighbourhood_radius+1):
    #       cell = spacetime[y_,x_]
    #       vector.append(cell)
    # for x_ in range(x-neighbourhood_radius,x+neighbourhood_radius+1):
    #   for y_ in range(y-neighbourhood_radius,y+neighbourhood_radius+1):
    #       cell = spacetime[y_,x_]
    #       vector.append(cell)
    return vector 

def complexity(spacetime:list[list[int]], x:int,y:int,r:int=1) -> float:
    vector = vectorise_neighbourhood(spacetime=spacetime, x=x, y=y, neighbourhood_radius=r)
    vector_str = ''.join(map(str,vector))
    vector_bytes = vector_str.encode('utf-8')
    total = len(vector_bytes)
    header = len(compress(''.encode('utf-8')))
    n = len(compress(vector_bytes))
    return (n-header)/total
    
def binarise_by_most_common_value(spacetime:ndarray) -> ndarray:
    #TODO: cutoff based on frequency - but below certain value not JUST the top one
    most_common_value = Counter(cell for row in spacetime.tolist() for cell in row).most_common(1)[0][0]
    return spacetime != most_common_value

def local_kolmogorov_complexity(spacetime:ndarray) -> ndarray:
    filtered = zeros_like(spacetime, dtype="float32")
    t,w = filtered.shape
    for y_ in range(1,t-1):
        for x_ in range(1,w-1):
            filtered[y_,x_] = complexity(
                spacetime=spacetime,
                x=x_,y=y_
                )
    imshow(filtered)
    show()
    return binarise_by_most_common_value(filtered)


from matplotlib.pyplot import imshow, show 
from eca import OneDimensionalElementaryCellularAutomata 

ca = OneDimensionalElementaryCellularAutomata(lattice_width=400)
for _ in range(400):
    ca.transition(110)

spacetime = ca.evolution()

imshow(spacetime)
show()
filtered = local_kolmogorov_complexity(spacetime)
imshow(filtered)
show()
