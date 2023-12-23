#include <stdio.h>


typedef char* (*test_function)();


struct test
{
	test_function func;
	const char* name;
};


char* test_dense_tensor_trace();
char* test_dense_tensor_transpose();
char* test_dense_tensor_dot();
char* test_dense_tensor_dot_update();
char* test_dense_tensor_kronecker_product();
char* test_dense_tensor_block();
char* test_block_sparse_tensor_copy();
char* test_block_sparse_tensor_get_block();
char* test_block_sparse_tensor_transpose();
char* test_block_sparse_tensor_reshape();
char* test_block_sparse_tensor_dot();
char* test_mps_to_statevector();


int main()
{
	struct test tests[] = {
		{ .func = test_dense_tensor_trace,             .name = "test_dense_tensor_trace" },
		{ .func = test_dense_tensor_transpose,         .name = "test_dense_tensor_transpose" },
		{ .func = test_dense_tensor_dot,               .name = "test_dense_tensor_dot" },
		{ .func = test_dense_tensor_dot_update,        .name = "test_dense_tensor_dot_update" },
		{ .func = test_dense_tensor_kronecker_product, .name = "test_dense_tensor_kronecker_product" },
		{ .func = test_dense_tensor_block,             .name = "test_dense_tensor_block" },
		{ .func = test_block_sparse_tensor_copy,       .name = "test_block_sparse_tensor_copy" },
		{ .func = test_block_sparse_tensor_get_block,  .name = "test_block_sparse_tensor_get_block" },
		{ .func = test_block_sparse_tensor_transpose,  .name = "test_block_sparse_tensor_transpose" },
		{ .func = test_block_sparse_tensor_reshape,    .name = "test_block_sparse_tensor_reshape" },
		{ .func = test_block_sparse_tensor_dot,        .name = "test_block_sparse_tensor_dot" },
		{ .func = test_mps_to_statevector,             .name = "test_mps_to_statevector" },
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
