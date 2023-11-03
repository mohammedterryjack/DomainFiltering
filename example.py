from argparse import ArgumentParser

from eca import OneDimensionalElementaryCellularAutomata
from matplotlib.pyplot import show, subplots

from domain_filters.lftsf import LocalisedFourierTransformSelfFilter
from domain_filters.simple import SimpleDomainFilter
from domain_filters.contours_via_circles import detect_contours

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--rule", type=int, default=110)
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--ic", type=int, default=None)
    arguments = parser.parse_args()

    simple_domain_filter = SimpleDomainFilter()
    lftsf_domain_filter = LocalisedFourierTransformSelfFilter()
    ca = OneDimensionalElementaryCellularAutomata(
        lattice_width=arguments.width, initial_configuration=arguments.ic
    )
    for _ in range(arguments.height):
        ca.transition(rule_number=arguments.rule)

    spacetime = ca.evolution()
    filtered_spacetime1 = simple_domain_filter.classify_spacetime(spacetime=spacetime)
    filtered_spacetime2 = lftsf_domain_filter.classify_spacetime(spacetime=spacetime)
    filtered_spacetine3 = detect_contours(image=spacetime, neighbourhood_radius=4, threshold=0.2)
    _, canvas = subplots(1, 4)
    canvas[0].imshow(spacetime, cmap="gray")
    canvas[1].imshow(filtered_spacetime1, cmap="gray")
    canvas[2].imshow(filtered_spacetime2, cmap="gray")
    canvas[3].imshow(filtered_spacetime3, cmap="gray")
    show()
