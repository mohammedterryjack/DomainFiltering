from eca import OneDimensionalElementaryCellularAutomata
from matplotlib.pyplot import bar, show
from numpy import array, concatenate, mean, ndarray, std


def neighbourhood_frequency(
    spacetime_evolution: ndarray, neighbourhoods: list[str]
) -> None:
    frequencies = {neighbourhood: 0 for neighbourhood in neighbourhoods}
    for row in spacetime_evolution:
        for a, b, c in zip(row, row[1:], row[2:]):
            neighbourhood = f"{a}{b}{c}"
            frequencies[neighbourhood] += 1
    return frequencies


def relatively_high_frequencies(frequencies: list[int]) -> list[int]:
    return [
        freq for freq in frequencies if (freq > (mean(frequencies) + std(frequencies)))
    ]


def filter_spacetime(
    spacetime_evolution: ndarray, transition_rule: dict[str, int]
) -> ndarray:
    filtered_spacetime = []
    for row in spacetime_evolution:
        filtered_row = []
        for a, b, c in zip(
            concatenate([[row[-1]], row[:-1]]), row, concatenate([row[1:], [row[0]]])
        ):
            neighbourhood = f"{a}{b}{c}"
            cell = transition_rule[neighbourhood]
            filtered_row.append(cell)
        filtered_spacetime.append(array(filtered_row))
    return array(filtered_spacetime)


def filter_by_lookup_frequency(
    spacetime_evolution: ndarray, display: bool = False
) -> ndarray:
    neighbourhoods = ["111", "110", "101", "100", "011", "010", "001", "000"]
    frequencies = neighbourhood_frequency(
        spacetime_evolution=spacetime_evolution, neighbourhoods=neighbourhoods
    )
    high_freq = relatively_high_frequencies(frequencies=list(frequencies.values()))
    high_frequencies = {
        key: value for key, value in frequencies.items() if value in high_freq
    }
    filter_transition_table = {
        neighbourhood: int(neighbourhood in high_frequencies)
        for neighbourhood in neighbourhoods
    }
    filtered_spacetime = filter_spacetime(
        spacetime_evolution=spacetime_evolution, transition_rule=filter_transition_table
    )
    if display:
        print(frequencies)
        print(filter_transition_table)
        low_frequencies = {
            key: value for key, value in frequencies.items() if value not in high_freq
        }
        bar(low_frequencies.keys(), low_frequencies.values())
        bar(high_frequencies.keys(), high_frequencies.values())
        show()
    return filtered_spacetime
