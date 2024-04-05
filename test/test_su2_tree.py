import numpy as np
import h5py


def _su2_j3_range(j1: int, j2: int):
    """
    Range of the combined quantum number 'j3' for given 'j1' and 'j2'.

    Quantum numbers are represented times 2 to support half-integers without rounding errors.
    """
    return range(abs(j1 - j2), j1 + j2 + 1, 2)


def su2_tree_enumerate_charge_sectors_data():

    # manually enumerate charge sectors for the tree:
    #
    #        |
    #        |7
    #        /\
    #       /  \5
    #      /   /\
    #     /  8/  \
    #    /   /\   \
    #   /   /  \   \
    #  2   0    4   1
    #
    charge_sectors = []
    for j4 in (0, 4, 10):
        for j0 in (1, 3):
            for j1 in (0, 2, 6):
                for j8 in _su2_j3_range(j0, j4):
                    for j5 in _su2_j3_range(j8, j1):
                        for j2 in (3, 7):
                            for j7 in _su2_j3_range(j2, j5):
                                charge_sectors.append([j0, j1, j2, 0, j4, j5, 0, j7, j8])
    charge_sectors = np.array(charge_sectors)

    with h5py.File("data/test_su2_tree_enumerate_charge_sectors.hdf5", "w") as file:
        file["charge_sectors"] = charge_sectors


def main():
    su2_tree_enumerate_charge_sectors_data()


if __name__ == "__main__":
    main()
