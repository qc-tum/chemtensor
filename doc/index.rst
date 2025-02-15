Welcome to ChemTensor's documentation!
======================================

.. image:: https://github.com/qc-tum/chemtensor/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/qc-tum/chemtensor/actions/workflows/ci.yml

`ChemTensor <https://github.com/qc-tum/chemtensor>`_ is an efficient tensor network library for electronic structure calculations of chemical systems (typically small molecules). It is based on the matrix product state (MPS) and tree tensor network state (TTNS) formalism and variants of DMRG.

`ChemTensor <https://github.com/qc-tum/chemtensor>`_ is written in C for performance reasons. The file `basic_dmrg_fermi_hubbard.c <https://github.com/qc-tum/chemtensor/blob/main/examples/dmrg/basic_dmrg_fermi_hubbard.c>`_ contains a standalone demonstration of constructing the Fermi-Hubbard model Hamiltonian as MPO and running DMRG.

`ChemTensor <https://github.com/qc-tum/chemtensor>`_ also offers a Python 3 interface for more straightforward accessibility and experimentation.


Examples
--------
Examples using the Python interface:

.. toctree::
    :maxdepth: 1

    examples/op_chains_mpo
    examples/op_chains_ttno
    examples/basic_dmrg_fermi_hubbard
    examples/water_molecule
    examples/operator_average


Python API
----------

.. toctree::
    :maxdepth: 2

    python/python_api


C source code
-------------

.. toctree::
    :maxdepth: 2

    src/c_common

.. toctree::
    :maxdepth: 2

    src/c_tensor

.. toctree::
    :maxdepth: 2

    src/c_state

.. toctree::
    :maxdepth: 2

    src/c_operator

.. toctree::
    :maxdepth: 2

    src/c_algorithm

.. toctree::
    :maxdepth: 2

    src/c_util
