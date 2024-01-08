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
char* test_dense_tensor_kronecker_product_degree_zero();
char* test_dense_tensor_qr();
char* test_dense_tensor_block();
char* test_block_sparse_tensor_copy();
char* test_block_sparse_tensor_get_block();
char* test_block_sparse_tensor_transpose();
char* test_block_sparse_tensor_reshape();
char* test_block_sparse_tensor_dot();
char* test_block_sparse_tensor_qr();
char* test_mps_to_statevector();
char* test_queue();


#define TEST_FUNCTION_ENTRY(fname) { .func = fname, .name = #fname }


int main()
{
	struct test tests[] = {
		TEST_FUNCTION_ENTRY(test_dense_tensor_trace),
		TEST_FUNCTION_ENTRY(test_dense_tensor_transpose),
		TEST_FUNCTION_ENTRY(test_dense_tensor_dot),
		TEST_FUNCTION_ENTRY(test_dense_tensor_dot_update),
		TEST_FUNCTION_ENTRY(test_dense_tensor_kronecker_product),
		TEST_FUNCTION_ENTRY(test_dense_tensor_kronecker_product_degree_zero),
		TEST_FUNCTION_ENTRY(test_dense_tensor_qr),
		TEST_FUNCTION_ENTRY(test_dense_tensor_block),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_copy),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_get_block),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_transpose),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_reshape),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_dot),
		TEST_FUNCTION_ENTRY(test_block_sparse_tensor_qr),
		TEST_FUNCTION_ENTRY(test_mps_to_statevector),
		TEST_FUNCTION_ENTRY(test_queue),
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
