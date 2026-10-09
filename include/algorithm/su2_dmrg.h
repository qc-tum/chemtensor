/// \file su2_dmrg.h
/// \brief DMRG algorithm for SU(2) symmetric tensors.

#pragma once

#include "su2_mps.h"
#include "su2_mpo.h"


//________________________________________________________________________________________________________________________
///
/// \brief Options for running the single-site DMRG algorithm for SU(2) symmetric tensors.
///
struct su2_dmrg_singlesite_options
{
	double delta_en_tol;  //!< stopping criterion based on relative energy difference between sweeps: abs(delta_en / max(en, 1)) < delta_en_tol
	int max_num_sweeps;   //!< maximum number of sweeps
	int maxiter_lanczos;  //!< maximum number of Lanczos iterations
};

void su2_dmrg_singlesite_default_options(struct su2_dmrg_singlesite_options* options);


//________________________________________________________________________________________________________________________
///
/// \brief Statistical information obtained while running the single-site DMRG algorithm for SU(2) symmetric tensors.
///
struct su2_dmrg_singlesite_info
{
	double* en_sweeps;  //!< energy after each sweep
	int num_sweeps;     //!< number of sweeps
};

void delete_su2_dmrg_singlesite_info(struct su2_dmrg_singlesite_info* info);


int su2_dmrg_singlesite(const struct su2_mpo* hamiltonian, const struct su2_dmrg_singlesite_options* options, struct su2_mps* psi, struct su2_dmrg_singlesite_info* info);



//________________________________________________________________________________________________________________________
///
/// \brief Options for running the two-site DMRG algorithm for SU(2) symmetric tensors.
///
struct su2_dmrg_twosite_options
{
	double delta_en_tol;  //!< stopping criterion based on relative energy difference between sweeps: abs(delta_en / max(en, 1)) < delta_en_tol
	int max_num_sweeps;   //!< maximum number of sweeps
	int maxiter_lanczos;  //!< maximum number of Lanczos iterations
	double tol_split;     //!< tolerance for splitting a temporarily merged MPS tensor pair
	ct_long max_vdim;     //!< maximum virtual bond dimension
};

void su2_dmrg_twosite_default_options(struct su2_dmrg_twosite_options* options);


//________________________________________________________________________________________________________________________
///
/// \brief Statistical information obtained while running the two-site DMRG algorithm for SU(2) symmetric tensors.
///
struct su2_dmrg_twosite_info
{
	double* entropy;    //!< von Neumann entropy of each (inner) virtual bond, at the last sweep
	double* en_sweeps;  //!< energy after each sweep
	int num_sweeps;     //!< number of sweeps
};

void delete_su2_dmrg_twosite_info(struct su2_dmrg_twosite_info* info);


int su2_dmrg_twosite(const struct su2_mpo* hamiltonian, const struct su2_dmrg_twosite_options* options, struct su2_mps* psi, struct su2_dmrg_twosite_info* info);
