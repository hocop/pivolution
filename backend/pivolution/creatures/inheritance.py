import numpy as np
from pydantic import BaseModel


N_CROSSOVER_POINTS = 3


def get_crossover_mask(length: int, n_points: int):
    if np.random.randint(2) == 0:
        mask = np.ones(length, dtype=bool)
    else:
        mask = np.zeros(length, dtype=bool)
    points = np.sort(np.random.randint(length, size=n_points + n_points % 2))
    if n_points % 2:
        points[-1] = length
    for i in range(0, n_points, 2):
        mask[points[i]: points[i + 1]] = ~mask[points[i]: points[i + 1]]
    return mask


def connected_components_1d(mask) -> list[tuple[int, int]]:
    start = None
    ccs = []
    for i, val in enumerate(mask):
        if start is None:
            if val:
                start = i
        else:
            if not val:
                ccs.append((start, i))
                start = None
    if val:
        ccs.append((start, len(mask)))
    return ccs


class Genome(BaseModel):
    dom_chromosome: np.ndarray
    rec_chromosome: np.ndarray

    def get_blended_choromosome(self) -> tuple[np.ndarray, np.ndarray]:
        crossover = get_crossover_mask(len(self.dom_chromosome), N_CROSSOVER_POINTS)
        chromosome = np.where(crossover, self.dom_chromosome, self.rec_chromosome)
        return chromosome, crossover

    def __str__(self):
        s = '\n'.join([
            ''.join(map(str, ch))
            for ch in [self.dom_chromosome, self.rec_chromosome]
        ])
        return s

    class Config:
        arbitrary_types_allowed = True


def get_child(parent_1: Genome, parent_2: Genome) -> Genome:
    chr_1, dom_1 = parent_1.get_blended_choromosome()
    chr_2, dom_2 = parent_2.get_blended_choromosome()

    dom_1_fixed = dom_1.copy()
    for start, end in connected_components_1d(dom_1 == dom_2):
        dom_1_fixed[start: end] = np.random.randint(2)

    dom_chromosome = np.where(dom_1_fixed, chr_1, chr_2)
    rec_chromosome = np.where(dom_1_fixed, chr_2, chr_1)

    return Genome(dom_chromosome=dom_chromosome, rec_chromosome=rec_chromosome)


if __name__ == '__main__':
    g_1 = Genome(dom_chromosome=np.array(['A'] * 100), rec_chromosome=np.array(['a'] * 100))
    g_2 = Genome(dom_chromosome=np.array(['B'] * 100), rec_chromosome=np.array(['b'] * 100))

    print(str(get_child(g_1, g_2)))
