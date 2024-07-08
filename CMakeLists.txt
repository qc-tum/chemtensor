
cmake_minimum_required(VERSION 3.8)

project(chemtensor)

include(CTest)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

set(CHEMTENSOR_DIRS "src" "src/tensor" "src/mps" "src/operator" "src/algorithm" "src/util")
set(CHEMTENSOR_SOURCES "src/tensor/dense_tensor.c" "src/tensor/block_sparse_tensor.c" "src/tensor/qnumber.c" "src/tensor/clebsch_gordan.c" "src/tensor/su2_recoupling.c" "src/tensor/su2_tree.c" "src/tensor/su2_tensor.c" "src/mps/mps.c" "src/operator/op_chain.c" "src/operator/local_op.c" "src/operator/mpo_graph.c" "src/operator/mpo.c" "src/operator/ttno_graph.c" "src/operator/ttno.c" "src/operator/hamiltonian.c" "src/algorithm/bond_ops.c" "src/algorithm/operation.c" "src/algorithm/dmrg.c" "src/algorithm/gradient.c" "src/util/util.c" "src/util/queue.c" "src/util/linked_list.c" "src/util/hash_table.c" "src/util/abstract_graph.c" "src/util/bipartite_graph.c" "src/util/integer_linear_algebra.c" "src/util/krylov.c" "src/util/pcg_basic.c" "src/util/rng.c")
set(TEST_SOURCES "test/tensor/test_dense_tensor.c" "test/tensor/test_block_sparse_tensor.c" "test/tensor/test_clebsch_gordan.c" "test/tensor/test_su2_tree.c" "test/tensor/test_su2_tensor.c" "test/mps/test_mps.c" "test/operator/test_mpo_graph.c" "test/operator/test_mpo.c" "test/operator/test_ttno_graph.c" "test/operator/test_ttno.c" "test/operator/test_hamiltonian.c" "test/algorithm/test_bond_ops.c" "test/algorithm/test_operation.c" "test/algorithm/test_dmrg.c" "test/algorithm/numerical_gradient.c" "test/algorithm/test_gradient.c" "test/util/test_queue.c" "test/util/test_linked_list.c" "test/util/test_hash_table.c" "test/util/test_bipartite_graph.c" "test/util/test_integer_linear_algebra.c" "test/util/test_krylov.c" "test/run_tests.c")

add_executable(chemtensor_test ${CHEMTENSOR_SOURCES} ${TEST_SOURCES})

target_include_directories(chemtensor_test PRIVATE ${CHEMTENSOR_DIRS} ${BLAS_INCLUDE_DIRS} ${LAPACKE_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
target_link_libraries(     chemtensor_test PRIVATE ${BLAS_LIBRARIES} ${LAPACKE_LIBRARIES} lapacke ${HDF5_LIBRARIES})

add_test(NAME chemtensor_test COMMAND chemtensor_test)
