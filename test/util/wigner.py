import numpy as np
from scipy.linalg import expm


def comm(a, b):
    """
    Commutator [a, b] = a b - b a.
    """
    return a @ b - b @ a


def construct_spin_operators(j: int):
    """
    Construct the spin operators S_x, S_y, S_z for quantum number 'j'.
    'j' is represented times 2 to retain the integer type.
    The 'm' quantum numbers are enumerated as -j, ..., j.
    """
    assert j >= 0

    # 'm' quantum numbers
    mlist = [0.5 * m2 for m2 in range(-j, j + 1, 2)]

    sup = np.diag([np.sqrt(0.5 * j * (0.5 * j + 1) - m * (m + 1)) for m in mlist[:-1]], k=-1)
    sdn = sup.T
    sx =  0.5  * (sup + sdn)
    sy = -0.5j * (sup - sdn)
    sz = np.diag(mlist)

    # check commutation relations
    assert np.allclose(comm(sx, sy), 1j * sz)
    assert np.allclose(comm(sy, sz), 1j * sx)
    assert np.allclose(comm(sz, sx), 1j * sy)

    return sx, sy, sz


def wigner_small_d(j: int, theta: float):
    """"
    Construct the Wigner "small" d-matrix for quantum number 'j' and rotation angle 'theta'.
    """
    _, sy, _ = construct_spin_operators(j)
    # imaginary unit cancels out
    return expm(-1j * theta * sy).real


def wigner_d(j: int, psi: float, theta: float, phi: float):
    """"
    Construct the Wigner D-matrix for quantum number 'j' and
    rotation angles 'psi', 'theta', 'phi' (Euler z-y-z rotation convention).
    """
    _, _, sz = construct_spin_operators(j)
    return expm(-1j * psi * sz) @ wigner_small_d(j, theta) @ expm(-1j * phi * sz)


def _delta(x, y):
    """
    Kronecker delta function.
    """
    return 1 if x == y else 0


def real_wigner_d(j: int, psi: float, theta: float, phi: float):
    """"
    Construct the real-valued Wigner D-matrix for quantum number 'j' and
    rotation angles 'psi', 'theta', 'phi' (Euler z-y-z rotation convention).
    The real form is designed so that multiplication with the spherical harmonics
    corresponds to an Euler rotation of the unit vector that parametrizes them.
    The 'm' quantum numbers are enumerated as -j, ..., j.
    'j' is the logical quantum number times 2, and must be even.
    """
    assert j >= 0
    # 'j' must be even
    assert j % 2 == 0

    # 'm' quantum numbers
    mlist = [m2 // 2 for m2 in range(-j, j + 1, 2)]

    # originally constructed for the real-valued spherical harmonics evaluated at (theta, -phi),
    # such that their theta- and phi-rotation direction matches the Wigner D-matrix rotation
    base_change = np.array([[
            1 / np.sqrt(2) * ((-1)**m * _delta(m, n) + _delta(-m, n)) if m > 0
            else (_delta(m, n) if m == 0
            else -1j / np.sqrt(2) * (_delta(m, n) - (-1)**m * _delta(-m, n)))
        for n in mlist]
        for m in mlist])

    w = base_change @ wigner_d(j, psi, theta, phi) @ base_change.conj().T
    assert np.allclose(w.imag, 0)

    return w.real
