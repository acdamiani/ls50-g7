import numpy as np
import numpy.typing as npt
from typing import Callable, Union, Literal


MutatorKind = Literal["constant"] | Literal["poisson"]


class ConstantMutator:
    _max_pop: int
    _dat: npt.NDArray[np.int64]
    _fitness_cost: float
    _epistasis: float
    _mutation_rate: float
    _gta_rate: float

    def __init__(
        self,
        population: int,
        fitness_cost: float = 0.1,
        mutation_rate: float = 0.03,
        epistasis: float = 1.0,
        gta_rate: float = 0.02,
    ) -> None:
        self._max_pop = population
        self._dat = np.zeros(population, dtype=np.int64)
        self._fitness_cost = fitness_cost
        self._epistasis = epistasis
        self._mutation_rate = mutation_rate
        self._gta_rate = gta_rate

    @property
    def population(self) -> int:
        return self._dat.shape[0]

    def select(self, mask: npt.NDArray[np.bool]) -> None:
        self._dat = self._dat[mask]

    def fitness(self) -> npt.NDArray[np.floating]:
        exp = np.pow(self._dat, self._epistasis)
        return np.pow(1.0 - self._fitness_cost, exp)

    def mutate(self) -> None:
        self._dat += np.random.poisson(self._mutation_rate, self.population)

    def replicate(self) -> None:
        np.random.shuffle(self._dat)
        end = min(self._max_pop, self.population * 2)
        self._dat = np.append(self._dat, self._dat)[:end]

    def horizontal_transfer(self) -> None:
        hits = np.random.rand(self.population) < self._gta_rate
        hit_ct = np.sum(hits)

        if hit_ct == 0:
            return

        targets = np.random.randint(0, self.population, size=hit_ct)
        self._dat[targets] = self._dat[hits]

        self.select(np.invert(hits))


class VariableMutator:
    _max_pop: int
    _dat: npt.NDArray[np.floating]
    _mean_fitness_cost: float
    _epistasis: float
    _alpha: float
    _beta: float
    _mutation_rate: float
    _gta_rate: float

    def __init__(
        self,
        population: int,
        alpha: float = 1.5,
        beta: float = 6.5,
        mutation_rate: float = 0.03,
        epistasis: float = 1.0,
        gta_rate: float = 0.02,
    ) -> None:
        self._max_pop = population
        self._dat = np.zeros((population, 0))
        self._epistasis = epistasis
        self._alpha = alpha
        self._beta = beta
        self._mutation_rate = mutation_rate
        self._gta_rate = gta_rate

    @property
    def population(self) -> int:
        return self._dat.shape[0]

    def select(self, mask: npt.NDArray[np.bool]) -> None:
        self._dat = self._dat[mask]

    def fitness(self) -> npt.NDArray[np.floating]:
        return np.pow(np.prod(1.0 - self._dat), self._epistasis)

    def mutate(self) -> None:
        cell_ct = self._dat.shape[0]

        new_mut = np.random.poisson(self._mutation_rate, size=cell_ct)
        mut_ct = np.max(new_mut)
        if mut_ct == 0:
            return

        costs = np.zeros((cell_ct, mut_ct), dtype=np.floating)
        for cell in range(cell_ct):
            cell_mut = new_mut[cell]
            costs[cell][:cell_mut] = np.random.beta(
                self._alpha, self._beta, size=cell_mut
            )

        self._dat = np.column_stack((self._dat, costs))

    def replicate(self) -> None:
        np.random.shuffle(self._dat)
        end = min(self._max_pop, self.population * 2)
        self._dat = np.append(self._dat, self._dat, axis=0)[:end]

    def horizontal_transfer(self) -> None:
        if self._gta_rate <= 0.0:
            return

        hits = np.random.rand(self.population) < self._gta_rate
        hit_ct = np.sum(hits)

        if hit_ct == 0:
            return

        targets = np.random.randint(0, self.population, size=hit_ct)
        self._dat[targets] = self._dat[hits]

        self.select(np.invert(hits))


class Culture:
    _fitness_cost: Callable[[int], npt.NDArray[np.long]]
    _gta_rate: float
    _epistasis: float
    _mutation_rate: float
    _mutator: Union[ConstantMutator, VariableMutator]

    def __init__(
        self,
        mutator: Union[ConstantMutator, VariableMutator],
    ) -> None:
        self._mutator = mutator

    @property
    def population(self) -> int:
        return self._mutator.population

    def fitness(self) -> npt.NDArray[np.floating]:
        return self._mutator.fitness()

    def mutate(self) -> None:
        self._mutator.mutate()
        self._mutator.horizontal_transfer()

        fitness = self._mutator.fitness()
        scythe = np.random.uniform(size=self._mutator.population)  # grim reaper

        self._mutator.select(scythe < fitness)  # selection
        self._mutator.replicate()  # replication
