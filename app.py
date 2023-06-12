from eca import OneDimensionalElementaryCellularAutomata
from numpy import ndarray, uint8
from PIL import Image
from streamlit import cache_data, image, number_input, set_page_config, slider, tabs

from domain_filters.lftsf import LocalisedFourierTransformSelfFilter
from domain_filters.simple import SimpleDomainFilter


@cache_data
def display_spacetime_as_image(spacetime: ndarray) -> None:
    image(Image.fromarray(uint8(spacetime) * 255))


@cache_data
def display_filtered_spacetime(spacetime: ndarray, simple: bool) -> None:
    if simple:
        filtered_spacetime = simple_domain_filter.classify_spacetime(
            spacetime=spacetime
        )
    else:
        filtered_spacetime = lftsf_domain_filter.classify_spacetime(spacetime=spacetime)
    display_spacetime_as_image(spacetime=filtered_spacetime)


set_page_config(
    page_title=f"Elementary Cellular Automata Domain Detection",
    page_icon="👾",
    initial_sidebar_state="expanded",
)
width = slider("Width", 10, 1000, 300)
height = slider("Height", 10, 1000, 300)
rule = number_input("Rule", 0, 258, 110)
simple_domain_filter = SimpleDomainFilter()
lftsf_domain_filter = LocalisedFourierTransformSelfFilter()
ca = OneDimensionalElementaryCellularAutomata(lattice_width=width)
for _ in range(height):
    ca.transition(rule_number=rule)
spacetime = ca.evolution()

original_tab, simple_tab, fourier_tab = tabs(
    ["Original", "Simple Heuristic", "Fourier Transform Self Filter"]
)
with original_tab:
    display_spacetime_as_image(spacetime=spacetime)
with simple_tab:
    display_filtered_spacetime(spacetime=spacetime, simple=True)
with fourier_tab:
    display_filtered_spacetime(spacetime=spacetime, simple=False)
