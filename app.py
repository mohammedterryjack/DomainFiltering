from eca import OneDimensionalElementaryCellularAutomata
from numpy import ndarray, uint8
from PIL import Image
from streamlit import cache_data, image, number_input, set_page_config, slider, tabs

from domain_filters.lftsf import LocalisedFourierTransformSelfFilter
from domain_filters.simple import SimpleDomainFilter


@cache_data
def get_spacetime(width: int, height: int, rule: int) -> ndarray:
    ca = OneDimensionalElementaryCellularAutomata(lattice_width=width)
    for _ in range(height):
        ca.transition(rule_number=rule)
    return ca.evolution()


@cache_data
def display_spacetime_as_image(spacetime: ndarray) -> None:
    image(Image.fromarray(uint8(spacetime) * 255))


@cache_data
def display_filtered_spacetime_simple(spacetime: ndarray, radius: int) -> None:
    simple_domain_filter = SimpleDomainFilter(max_radius=radius)
    filtered_spacetime = simple_domain_filter.classify_spacetime(spacetime=spacetime)
    display_spacetime_as_image(spacetime=filtered_spacetime)


@cache_data
def display_filtered_spacetime_fourier(
    spacetime: ndarray, binarisation_threshold: float, localisation: int
) -> None:
    lftsf_domain_filter = LocalisedFourierTransformSelfFilter(
        localisation_size=localisation, binarisation_threshold=binarisation_threshold
    )
    filtered_spacetime = lftsf_domain_filter.classify_spacetime(spacetime=spacetime)
    display_spacetime_as_image(spacetime=filtered_spacetime)


set_page_config(
    page_title=f"Elementary Cellular Automata Domain Detection",
    page_icon="👾",
    initial_sidebar_state="expanded",
)
original_tab, simple_tab, fourier_tab = tabs(
    ["Original", "Simple Heuristic", "Fourier Transform Self Filter"]
)

width = slider("Width", 10, 1000, 300)
height = slider("Height", 10, 1000, 300)
rule = number_input("Rule", 0, 258, 110)
spacetime = get_spacetime(width=width, height=height, rule=rule)

with original_tab:
    display_spacetime_as_image(spacetime=spacetime)
with simple_tab:
    radius = slider("Max Radius", 2, width // 2, 15)
    display_filtered_spacetime_simple(spacetime=spacetime, radius=radius)
with fourier_tab:
    binarisation_threshold = slider("Binarisation Threshold", 0.0, 1.0, 0.5)
    localisation = slider("Submatrix Size", 2, width // 2, 4)
    display_filtered_spacetime_fourier(
        spacetime=spacetime,
        binarisation_threshold=binarisation_threshold,
        localisation=localisation,
    )
