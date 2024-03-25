#include <stdio.h>


typedef char* (*test_function)();


struct test
{
	test_function func;
	const char* name;
};


char* test_dense_tensor_trace();
char* test_dense_tensor_cyclic_partial_trace();
char* test_dense_tensor_transpose();
char* test_dense_tensor_slice();
char* test_dense_tensor_multiply_pointwise();
char* test_dense_tensor_dot();
char* test_dense_tensor_dot_update();
char* test_dense_tensor_kronecker_product();
char* test_dense_tensor_kronecker_product_degree_zero();
char* test_dense_tensor_qr();
char* test_dense_tensor_rq();
char* test_dense_tensor_svd();
char* test_dense_tensor_block();
char* test_block_sparse_tensor_copy();
char* test_block_sparse_tensor_get_block();
char* test_block_sparse_tensor_cyclic_partial_trace();
char* test_block_sparse_tensor_norm2();
char* test_block_sparse_tensor_transpose();
char* test_block_sparse_tensor_reshape();
char* test_block_sparse_tensor_slice();
char* test_block_sparse_tensor_multiply_pointwise_vector();
char* test_block_sparse_tensor_dot();
char* test_block_sparse_tensor_qr();
char* test_block_sparse_tensor_rq();
char* test_block_sparse_tensor_svd();
char* test_block_sparse_tensor_serialize();
char* test_mps_orthonormalize_qr();
char* test_mps_split_tensor_svd();
char* test_mps_to_statevector();
char* test_mps_vdot();
char* test_queue();
char* test_hash_table();
char* test_bipartite_graph_maximum_cardinality_matching();
char* test_bipartite_graph_minimum_vertex_cover();
char* test_lanczos_iteration_d();
char* test_lanczos_iteration_z();
char* test_eigensystem_krylov_symmetric();
char* test_eigensystem_krylov_hermitian();
char* test_mpo_graph_from_opchains_basic();
char* test_mpo_graph_from_opchains_advanced();
char* test_mpo_from_graph();
char* test_retained_bond_indices();
char* test_split_block_sparse_matrix_svd();
char* test_split_block_sparse_matrix_svd_zero();
char* test_operator_inner_product();


#define TEST_FUNCTION_ENTRY(fname) { .func = fname, .name = #fname }


int main()
{
	struct test tests[] = {
		TEST_FUNCTION_ENTRY(test_dense_tensor_trace),
		TEST_FUNCTION_ENTRY(test_dense_tensor_cyclic_partial_trace),
		TEST_FUNCTION_ENTRY(test_dense_tensor_transpose),
		TEST_FUNCTION_ENTRY(test_dense_tensor_slice),
		TEST_FUNCTION_ENTRY(test_dense_tensor_multiply_pointwise),
		TEST_FUNCTION_ENTRY(test_dense_tensor_dot),
		TEST_FUNCTION_ENTRY(test_dense_tensor_dot_update),
		TEST_FUNCTION_ENTRY(test_dense_tensor_kronecker_product),
		TEST_FUNCTION_ENTRY(test_dense_tensor_kronecker_product_degree_zero),
		TEST_FUNCTION_ENTRY(test_dense_tensor_qr),
		TEST_FUNCTION_ENTRY(test_dense_tensor_rq),
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
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_dot),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_qr),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_rq),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_svd),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_serialize),
		TEST_FUNCTION_ENTRY(test_mps_orthonormalize_qr),
		TEST_FUNCTION_ENTRY(test_mps_split_tensor_svd),
		TEST_FUNCTION_ENTRY(test_mps_to_statevector),
		TEST_FUNCTION_ENTRY(test_mps_vdot),
		TEST_FUNCTION_ENTRY(test_queue),
		TEST_FUNCTION_ENTRY(test_hash_table),
		TEST_FUNCTION_ENTRY(test_bipartite_graph_maximum_cardinality_matching),
		TEST_FUNCTION_ENTRY(test_bipartite_graph_minimum_vertex_cover),
		TEST_FUNCTION_ENTRY(test_lanczos_iteration_d),
		TEST_FUNCTION_ENTRY(test_lanczos_iteration_z),
		TEST_FUNCTION_ENTRY(test_eigensystem_krylov_symmetric),
		TEST_FUNCTION_ENTRY(test_eigensystem_krylov_hermitian),
		TEST_FUNCTION_ENTRY(test_mpo_graph_from_opchains_basic),
		TEST_FUNCTION_ENTRY(test_mpo_graph_from_opchains_advanced),
		TEST_FUNCTION_ENTRY(test_mpo_from_graph),
		TEST_FUNCTION_ENTRY(test_retained_bond_indices),
		TEST_FUNCTION_ENTRY(test_split_block_sparse_matrix_svd),
		TEST_FUNCTION_ENTRY(test_split_block_sparse_matrix_svd_zero),
		TEST_FUNCTION_ENTRY(test_operator_inner_product),
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
