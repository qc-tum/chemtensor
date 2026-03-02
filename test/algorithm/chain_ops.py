import numpy as np
from mps import MPS
from mpo import MPO


def contraction_operator_step_right(a: np.ndarray, b: np.ndarray, w: np.ndarray, r: np.ndarray):
    r"""
    Contraction step from right to left, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

       ╭───────╮       ╭─────────╮
       │       │       │         │
     ──0   b*  2──   ──2         │
       │       │       │         │
       ╰───1───╯       │         │
           │           │         │
                       │         │
           │           │         │
       ╭───1───╮       │         │
       │       │       │         │
     ──0   w   3──   ──1    r    │
       │       │       │         │
       ╰───2───╯       │         │
           │           │         │
                       │         │
           │           │         │
       ╭───1───╮       │         │
       │       │       │         │
     ──0   a   2──   ──0         │
       │       │       │         │
       ╰───────╯       ╰─────────╯
    """
    assert a.ndim == 3
    assert b.ndim == 3
    assert w.ndim == 4
    assert r.ndim == 3
    # multiply with 'a' tensor
    t = np.tensordot(a, r, 1)
    # multiply with 'w' tensor
    t = np.tensordot(w, t, axes=((2, 3), (1, 2)))
    # make original left virtual bond of 'a' the leading dimension
    t = t.transpose((2, 0, 1, 3))
    # multiply with conjugated 'b' tensor
    r_next = np.tensordot(t, b.conj(), axes=((2, 3), (1, 2)))
    return r_next


def contraction_operator_step_left(a: np.ndarray, b: np.ndarray, w: np.ndarray, l: np.ndarray):
    r"""
    Contraction step from left to right, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

     ╭─────────╮       ╭───────╮
     │         │       │       │
     │         2──   ──0   b*  2──
     │         │       │       │
     │         │       ╰───1───╯
     │         │           │
     │         │
     │         │           │
     │         │       ╭───1───╮
     │         │       │       │
     │    l    1──   ──0   w   3──
     │         │       │       │
     │         │       ╰───2───╯
     │         │           │
     │         │
     │         │           │
     │         │       ╭───1───╮
     │         │       │       │
     │         0──   ──0   a   2──
     │         │       │       │
     ╰─────────╯       ╰───────╯
    """
    assert a.ndim == 3
    assert b.ndim == 3
    assert w.ndim == 4
    assert l.ndim == 3
    # multiply with conjugated 'b' tensor
    t = np.tensordot(l, b.conj(), axes=(2, 0))
    # multiply with 'w' tensor
    t = np.tensordot(w, t, axes=((0, 1), (1, 2)))
    # multiply with 'a' tensor
    l_next = np.tensordot(a, t, axes=((0, 1), (2, 0)))
    return l_next


def compute_right_operator_blocks(psi: MPS, op: MPO):
    """
    Compute all partial contractions from the right.
    """
    nsites = psi.nsites
    assert nsites == op.nsites
    blocks = [None for _ in range(nsites)]
    # initialize rightmost dummy block
    blocks[nsites - 1] = np.array([[[1]]])
    for i in reversed(range(nsites - 1)):
        blocks[i] = contraction_operator_step_right(
            psi.a[i + 1], psi.a[i + 1], op.a[i + 1], blocks[i + 1])
    return blocks


def mpo_inner_product(chi: MPS, op: MPO, psi: MPS):
    """
    Compute the inner product `<chi | op | psi>`.

    Args:
        chi: wavefunction represented as MPS
        op:  operator represented as MPO
        psi: wavefunction represented as MPS

    Returns:
        `<chi | op | psi>`
    """
    assert chi.nsites == op.nsites
    assert psi.nsites == op.nsites
    if psi.nsites == 0:
        return 0
    # initialize 't' by the identity matrix
    assert chi.a[-1].shape[2] == psi.a[-1].shape[2]
    t = np.identity(psi.a[-1].shape[2], dtype=psi.a[-1].dtype)
    t = t.reshape((psi.a[-1].shape[2], 1, psi.a[-1].shape[2]))
    for i in reversed(range(psi.nsites)):
        t = contraction_operator_step_right(psi.a[i], chi.a[i], op.a[i], t)
    # t should now be a 1 x 1 x 1 tensor
    assert t.shape == (1, 1, 1)
    return t[0, 0, 0]


def apply_local_hamiltonian(a: np.ndarray, w: np.ndarray, l: np.ndarray, r: np.ndarray):
    r"""
    Apply a local Hamiltonian operator.

    To-be contracted tensor network (the indices at the open legs
    show the ordering for the output tensor)::

           .................................
          '                                 '
     ╭────:────╮                       ╭────:────╮
     │    :    │                       │    :    │
     │    :    2──   0           2   ──2    :    │
     │    :    │                       │    :    │
     │    :    │                       │    :    │
     │    :    │           1           │    :    │
     │    :    │                       │    :    │
     │    :    │           │           │    :    │
     │    '....│.......╭───1───╮.......│....'    │
     │         │       │       │       │         │
     │    l    1──   ──0   w   3──   ──1    r    │
     │         │       │       │       │         │
     │         │       ╰───2───╯       │         │
     │         │           │           │         │
     │         │                       │         │
     │         │           │           │         │
     │         │       ╭───1───╮       │         │
     │         │       │       │       │         │
     │         0──   ──0   a   2──   ──0         │
     │         │       │       │       │         │
     ╰─────────╯       ╰───────╯       ╰─────────╯
    """
    assert a.ndim == 3
    assert w.ndim == 4
    assert l.ndim == 3
    assert r.ndim == 3
    # multiply 'a' with 'r' tensor and store result in 't'
    t = np.tensordot(a, r, 1)
    # multiply 't' with 'w' tensor
    # multiply with 'w' tensor
    t = np.tensordot(w, t, axes=((2, 3), (1, 2)))
    # multiply 't' with 'l' tensor
    t = np.tensordot(l, t, axes=((0, 1), (2, 0)))
    return t
