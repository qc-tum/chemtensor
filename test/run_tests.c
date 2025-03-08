#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif


typedef char* (*test_function)();


struct test
{
	test_function func;
	const char* name;
};


char* test_tensor_index_to_offset();
char* test_dense_tensor_trace();
char* test_dense_tensor_cyclic_partial_trace();
char* test_dense_tensor_transpose();
char* test_dense_tensor_slice();
char* test_dense_tensor_multiply_pointwise();
char* test_dense_tensor_multiply_axis();
char* test_dense_tensor_dot();
char* test_dense_tensor_dot_update();
char* test_dense_tensor_kronecker_product();
char* test_dense_tensor_kronecker_product_degree_zero();
char* test_dense_tensor_concatenate();
char* test_dense_tensor_block_diag();
char* test_dense_tensor_qr();
char* test_dense_tensor_rq();
char* test_dense_tensor_eigh();
char* test_dense_tensor_svd();
char* test_dense_tensor_block();
char* test_block_sparse_tensor_copy();
char* test_block_sparse_tensor_get_block();
char* test_block_sparse_tensor_cyclic_partial_trace();
char* test_block_sparse_tensor_norm2();
char* test_block_sparse_tensor_transpose();
char* test_block_sparse_tensor_reshape();
char* test_block_sparse_tensor_slice();
char* test_dense_tensor_pad_zeros();
char* test_block_sparse_tensor_multiply_pointwise_vector();
char* test_block_sparse_tensor_multiply_axis();
char* test_block_sparse_tensor_dot();
char* test_block_sparse_tensor_concatenate();
char* test_block_sparse_tensor_block_diag();
char* test_block_sparse_tensor_qr();
char* test_block_sparse_tensor_rq();
char* test_block_sparse_tensor_svd();
char* test_block_sparse_tensor_serialize();
char* test_block_sparse_tensor_get_entry();
char* test_clebsch_gordan_coefficients();
char* test_su2_tree_enumerate_charge_sectors();
char* test_su2_fuse_split_tree_enumerate_charge_sectors();
char* test_su2_graph_to_fuse_split_tree();
char* test_su2_graph_yoga_to_simple_subtree();
char* test_su2_graph_connect();
char* test_su2_tensor_fmove();
char* test_su2_tensor_fuse_axes();
char* test_su2_tensor_split_axis();
char* test_su2_tensor_contract_simple();
char* test_su2_tensor_contract_yoga();
char* test_su2_to_dense_tensor();
char* test_copy_mps();
char* test_mps_vdot();
char* test_mps_add();
char* test_mps_orthonormalize_qr();
char* test_mps_compress();
char* test_mps_split_tensor_svd();
char* test_mps_to_statevector();
char* test_save_mps();
char* test_ttns_vdot();
char* test_queue();
char* test_linked_list();
char* test_hash_table();
char* test_bipartite_graph_maximum_cardinality_matching();
char* test_bipartite_graph_minimum_vertex_cover();
char* test_integer_hermite_normal_form();
char* test_integer_backsubstitute();
char* test_lanczos_iteration_d();
char* test_lanczos_iteration_z();
char* test_eigensystem_krylov_symmetric();
char* test_eigensystem_krylov_hermitian();
char* test_mpo_graph_from_opchains_basic();
char* test_mpo_graph_from_opchains_advanced();
char* test_mpo_from_assembly();
char* test_ttno_graph_from_opchains();
char* test_ttno_from_assembly();
char* test_ising_1d_mpo();
char* test_heisenberg_xxz_1d_mpo();
char* test_bose_hubbard_1d_mpo();
char* test_fermi_hubbard_1d_mpo();
char* test_molecular_hamiltonian_mpo();
char* test_spin_molecular_hamiltonian_mpo();
char* test_quadratic_fermionic_mpo();
char* test_quadratic_spin_fermionic_mpo();
char* test_retained_bond_indices();
char* test_split_block_sparse_matrix_svd();
char* test_split_block_sparse_matrix_svd_zero();
char* test_mpo_inner_product();
char* test_apply_mpo();
char* test_ttno_inner_product();
char* test_dmrg_singlesite();
char* test_dmrg_twosite();
char* test_operator_average_coefficient_gradient();
char* test_apply_thc_spin_molecular_hamiltonian();
char* test_thc_spin_molecular_hamiltonian_to_matrix();


#define TEST_FUNCTION_ENTRY(fname) { .func = fname, .name = #fname }


int main()
{
	#ifdef _OPENMP
	printf("maximum number of OpenMP threads: %d\n", omp_get_max_threads());
	#else
	printf("OpenMP not available\n");
	#endif

	struct test tests[] = {
		TEST_FUNCTION_ENTRY(test_tensor_index_to_offset),
		TEST_FUNCTION_ENTRY(test_dense_tensor_trace),
		TEST_FUNCTION_ENTRY(test_dense_tensor_cyclic_partial_trace),
		TEST_FUNCTION_ENTRY(test_dense_tensor_transpose),
		TEST_FUNCTION_ENTRY(test_dense_tensor_slice),
		TEST_FUNCTION_ENTRY(test_dense_tensor_pad_zeros),
		TEST_FUNCTION_ENTRY(test_dense_tensor_multiply_pointwise),
		TEST_FUNCTION_ENTRY(test_dense_tensor_multiply_axis),
		TEST_FUNCTION_ENTRY(test_dense_tensor_dot),
		TEST_FUNCTION_ENTRY(test_dense_tensor_dot_update),
		TEST_FUNCTION_ENTRY(test_dense_tensor_kronecker_product),
		TEST_FUNCTION_ENTRY(test_dense_tensor_kronecker_product_degree_zero),
		TEST_FUNCTION_ENTRY(test_dense_tensor_concatenate),
		TEST_FUNCTION_ENTRY(test_dense_tensor_block_diag),
		TEST_FUNCTION_ENTRY(test_dense_tensor_qr),
		TEST_FUNCTION_ENTRY(test_dense_tensor_rq),
		TEST_FUNCTION_ENTRY(test_dense_tensor_eigh),
		TEST_FUNCTION_ENTRY(test_dense_tensor_svd),
		TEST_FUNCTION_ENTRY(test_dense_tensor_block),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_copy),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_get_block),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_cyclic_partial_trace),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_norm2),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_transpose),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_reshape),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_slice),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_multiply_pointwise_vector),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_multiply_axis),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_dot),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_concatenate),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_block_diag),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_qr),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_rq),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_svd),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_serialize),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_get_entry),
		TEST_FUNCTION_ENTRY(test_clebsch_gordan_coefficients),
		TEST_FUNCTION_ENTRY(test_su2_tree_enumerate_charge_sectors),
		TEST_FUNCTION_ENTRY(test_su2_fuse_split_tree_enumerate_charge_sectors),
		TEST_FUNCTION_ENTRY(test_su2_graph_to_fuse_split_tree),
		TEST_FUNCTION_ENTRY(test_su2_graph_yoga_to_simple_subtree),
		TEST_FUNCTION_ENTRY(test_su2_graph_connect),
		TEST_FUNCTION_ENTRY(test_su2_tensor_fmove),
		TEST_FUNCTION_ENTRY(test_su2_tensor_fuse_axes),
		TEST_FUNCTION_ENTRY(test_su2_tensor_split_axis),
		TEST_FUNCTION_ENTRY(test_su2_tensor_contract_simple),
		TEST_FUNCTION_ENTRY(test_su2_tensor_contract_yoga),
		TEST_FUNCTION_ENTRY(test_su2_to_dense_tensor),
		TEST_FUNCTION_ENTRY(test_copy_mps),
		TEST_FUNCTION_ENTRY(test_mps_vdot),
		TEST_FUNCTION_ENTRY(test_mps_add),
		TEST_FUNCTION_ENTRY(test_mps_orthonormalize_qr),
		TEST_FUNCTION_ENTRY(test_mps_compress),
		TEST_FUNCTION_ENTRY(test_mps_split_tensor_svd),
		TEST_FUNCTION_ENTRY(test_mps_to_statevector),
		TEST_FUNCTION_ENTRY(test_save_mps),
		TEST_FUNCTION_ENTRY(test_ttns_vdot),
		TEST_FUNCTION_ENTRY(test_queue),
		TEST_FUNCTION_ENTRY(test_linked_list),
		TEST_FUNCTION_ENTRY(test_hash_table),
		TEST_FUNCTION_ENTRY(test_bipartite_graph_maximum_cardinality_matching),
		TEST_FUNCTION_ENTRY(test_bipartite_graph_minimum_vertex_cover),
		TEST_FUNCTION_ENTRY(test_integer_hermite_normal_form),
		TEST_FUNCTION_ENTRY(test_integer_backsubstitute),
		TEST_FUNCTION_ENTRY(test_lanczos_iteration_d),
		TEST_FUNCTION_ENTRY(test_lanczos_iteration_z),
		TEST_FUNCTION_ENTRY(test_eigensystem_krylov_symmetric),
		TEST_FUNCTION_ENTRY(test_eigensystem_krylov_hermitian),
		TEST_FUNCTION_ENTRY(test_mpo_graph_from_opchains_basic),
		TEST_FUNCTION_ENTRY(test_mpo_graph_from_opchains_advanced),
		TEST_FUNCTION_ENTRY(test_mpo_from_assembly),
		TEST_FUNCTION_ENTRY(test_ttno_graph_from_opchains),
		TEST_FUNCTION_ENTRY(test_ttno_from_assembly),
		TEST_FUNCTION_ENTRY(test_ising_1d_mpo),
		TEST_FUNCTION_ENTRY(test_heisenberg_xxz_1d_mpo),
		TEST_FUNCTION_ENTRY(test_bose_hubbard_1d_mpo),
		TEST_FUNCTION_ENTRY(test_fermi_hubbard_1d_mpo),
		TEST_FUNCTION_ENTRY(test_molecular_hamiltonian_mpo),
		TEST_FUNCTION_ENTRY(test_spin_molecular_hamiltonian_mpo),
		TEST_FUNCTION_ENTRY(test_quadratic_fermionic_mpo),
		TEST_FUNCTION_ENTRY(test_quadratic_spin_fermionic_mpo),
		TEST_FUNCTION_ENTRY(test_retained_bond_indices),
		TEST_FUNCTION_ENTRY(test_split_block_sparse_matrix_svd),
		TEST_FUNCTION_ENTRY(test_split_block_sparse_matrix_svd_zero),
		TEST_FUNCTION_ENTRY(test_mpo_inner_product),
		TEST_FUNCTION_ENTRY(test_apply_mpo),
		TEST_FUNCTION_ENTRY(test_ttno_inner_product),
		TEST_FUNCTION_ENTRY(test_dmrg_singlesite),
		TEST_FUNCTION_ENTRY(test_dmrg_twosite),
		TEST_FUNCTION_ENTRY(test_operator_average_coefficient_gradient),
		TEST_FUNCTION_ENTRY(test_apply_thc_spin_molecular_hamiltonian),
		TEST_FUNCTION_ENTRY(test_thc_spin_molecular_hamiltonian_to_matrix),
	};
	int num_tests = sizeof(tests) / sizeof(struct test);

	int num_pass = 0;
	for (int i = 0; i < num_tests; i++)
	{
		printf(".");
		char* msg = tests[i].func();
		if (msg == 0) {
			num_pass++;
		}
		else {
			printf("\nTest '%s' failed: %s\n", tests[i].name, msg);
		}
	}
	printf("\nNumber of successful tests: %i / %i\n", num_pass, num_tests);

	if (num_pass < num_tests)
	{
		printf("At least one test failed!\n");
	}
	else
	{
		printf("All tests passed.\n");
	}

	return num_pass != num_tests;
}
