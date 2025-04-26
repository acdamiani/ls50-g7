import numpy as np
import numpy.typing as npt


class Generation:
    population: int
    size: int

    _mutations: npt.NDArray[np.int64]
    _fitness_cost: float
    _epistasis: float
    _mutation_rate: float

    def __init__(
        self,
        population: int,
        size: int,
        mutations: npt.NDArray[np.int64] | None = None,
        fitness_cost=0.1,
        epistasis=1.0,
        mutation_rate=1.0,
    ) -> None:
        self.population = population
        self.size = size
        self._mutations = (
            np.zeros(population, dtype=np.int64) if not mutations else mutations
        )
        self._epistasis = epistasis
        self._fitness_cost = fitness_cost
        self._mutation_rate = mutation_rate

    def mutate(self) -> None:
        mutations = self._mutations + np.random.poisson(
            self._mutation_rate, size=self.population
        )
        fitness = self._fitness(mutations)
        scythe = np.random.uniform(size=self.population)  # grim reaper

        population = mutations[np.nonzero(scythe < fitness)]  # selection
        size = min(population.shape[0] * 2, self.population)

        self.population = size
        self._mutations = np.append(population, population)[:size]  # reproduction

    def _fitness(self, mutations) -> npt.NDArray[np.float32]:
        exp = np.pow(mutations, self._epistasis)
        return np.pow(1.0 - self._fitness_cost, exp)

    def fitness(self) -> npt.NDArray[np.float32]:
        return self._fitness(self._mutations)
