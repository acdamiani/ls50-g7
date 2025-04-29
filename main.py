from typing import Literal, Sequence
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from culture import ConstantMutator, Culture, VariableMutator
from concurrent.futures import ProcessPoolExecutor, as_completed

GENERATIONS = 100
POPULATION = 1_000_000


def param_desc(args: dict[str, float]):
    return ";".join(f"{k}={v:.2f}" for k, v in args.items())


def run_sim(generations: int, mutator: VariableMutator | ConstantMutator, label: str):
    culture = Culture(mutator)
    fitness = np.empty(generations, dtype=np.float32)

    for gen in range(generations):
        fitness[gen] = np.mean(culture.fitness())
        culture.mutate()

    print(culture.population)

    return label, fitness


def fiddle(
    kind: Literal["constant"] | Literal["variable"],
    target: str,
    values: npt.NDArray[np.floating] | Sequence[float],
    generations=GENERATIONS,
    population=POPULATION,
    seed=None,
    **params: float,
):
    if seed is not None:
        np.random.seed(seed)

    constr = None
    if kind == "constant":
        constr = ConstantMutator
    elif kind == "variable":
        constr = VariableMutator

    if not constr:
        return

    mutators = [
        (params | {target: v}, constr(population, **(params | {target: v})))
        for v in values
    ]

    results: dict[str, npt.NDArray[np.float32]] = {}
    with ProcessPoolExecutor(max_workers=len(values)) as pool:
        futures = {
            pool.submit(run_sim, generations, mutator, str(params[target]))
            for params, mutator in mutators
        }

        for fut in as_completed(futures):
            label, series = fut.result()
            results[label] = series
    return results


def main():
    results = fiddle(
        "variable",
        "gta_rate",
        [0.0, 0.05, 0.1],
        mutation_rate=0.1,
    )

    if not results:
        return

    fig, ax = plt.subplots()
    t = np.arange(GENERATIONS)

    for info, series in sorted(results.items()):
        ax.plot(t, series, label=info, linestyle="dashed", alpha=0.5)

    ax.set(
        xlabel="Generation",
        ylabel="Mean fitness",
        title="Fitness across GTA rates",
    )
    ax.legend(frameon=False)
    fig.savefig("out.png")


if __name__ == "__main__":
    main()
