
cmake_minimum_required(VERSION 3.8)

project(chemtensor)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

set(CHEMTENSOR_DIRS "src" "src/tensor" "src/mps" "src/operator" "src/algorithm" "src/util")
set(CHEMTENSOR_SOURCES "src/tensor/dense_tensor.c" "src/tensor/block_sparse_tensor.c" "src/tensor/qnumber.c" "src/mps/mps.c" "src/operator/op_chain.c" "src/operator/mpo_graph.c" "src/operator/mpo.c" "src/algorithm/bond_ops.c" "src/util/util.c" "src/util/queue.c" "src/util/hash_table.c" "src/util/bipartite_graph.c")
set(TEST_SOURCES "test/test_dense_tensor.c" "test/test_block_sparse_tensor.c" "test/test_mps.c" "test/test_mpo_graph.c" "test/test_mpo.c" "test/test_bond_ops.c" "test/test_queue.c" "test/test_hash_table.c" "test/test_bipartite_graph.c" "test/run_tests.c")

add_executable(chemtensor_test ${CHEMTENSOR_SOURCES} ${TEST_SOURCES})

target_include_directories(chemtensor_test PRIVATE ${CHEMTENSOR_DIRS} ${BLAS_INCLUDE_DIRS} ${LAPACKE_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
target_link_libraries(     chemtensor_test PRIVATE ${BLAS_LIBRARIES} ${LAPACKE_LIBRARIES} lapacke ${HDF5_LIBRARIES})
