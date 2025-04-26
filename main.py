import numpy as np
import sys
from generation import Generation


def main():
    # create a genome
    generation = Generation(1_000_000, 12_000_000)

    with open("out.txt", "w") as f:
        last = None
        current = np.mean(generation.fitness())
        while True:
            generation.mutate()
            last = current
            current = np.mean(generation.fitness())
            f.write(f"{current}\n")


if __name__ == "__main__":
    main()
