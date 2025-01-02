/// \file thc.h
/// \brief Tensor hypercontraction (THC) representation of a molecular Hamiltonian and corresponding utility functions (see <a href="https://arxiv.org/abs/2409.12708">arXiv:2409.12708</a>).

#pragma once

#include "mps.h"
#include "mpo.h"


//________________________________________________________________________________________________________________________
///
/// \brief Tensor hypercontraction (THC) representation of a molecular Hamiltonian, assuming a spin orbital basis.
///
struct thc_spin_molecular_hamiltonian
{
	struct dense_tensor tkin;           //!< coefficients of the kinetic (one-body) term, must be symmetric
	struct dense_tensor thc_kernel;     //!< kernel matrix of the THC representation, must be symmetric
	struct dense_tensor thc_transform;  //!< transformation matrix (from kernel to orbital space) of the THC representation
	struct dense_tensor en_kin;         //!< eigenvalues of the kinetic coefficient matrix (real-valued)
	struct dense_tensor u_kin;          //!< eigenvectors of the kinetic coefficient matrix
	struct mpo* mpo_kin;                //!< elementary quadratic MPOs for the kinetic term
	struct mpo* mpo_thc;                //!< elementary quadratic MPOs for the interaction term in THC representation
};


int construct_thc_spin_molecular_hamiltonian(const struct dense_tensor* restrict tkin, const struct dense_tensor* restrict thc_kernel, const struct dense_tensor* restrict thc_transform, struct thc_spin_molecular_hamiltonian* hamiltonian);

void delete_thc_spin_molecular_hamiltonian(struct thc_spin_molecular_hamiltonian* hamiltonian);

//________________________________________________________________________________________________________________________
//

// application to a state in MPS form

int apply_thc_spin_molecular_hamiltonian(const struct thc_spin_molecular_hamiltonian* hamiltonian,
	const struct mps* restrict psi, const double tol, const long max_vdim, struct mps* restrict h_psi);

//________________________________________________________________________________________________________________________
//

// conversion to full matrix (intended for testing)

int thc_spin_molecular_hamiltonian_to_matrix(const struct thc_spin_molecular_hamiltonian* hamiltonian, struct block_sparse_tensor* mat);
