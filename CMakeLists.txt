
cmake_minimum_required(VERSION 3.8)

project(chemtensor)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

set(CHEMTENSOR_SOURCES "src/tensor/dense_tensor.c" "src/tensor/block_sparse_tensor.c" "src/util/util.c")
set(TEST_SOURCES "test/test_dense_tensor.c" "test/test_block_sparse_tensor.c" "test/run_tests.c")

add_executable(chemtensor_test ${CHEMTENSOR_SOURCES} ${TEST_SOURCES})

target_include_directories(chemtensor_test PRIVATE ${BLAS_INCLUDE_DIRS} ${LAPACKE_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src" "src/tensor" "src/util")
target_link_libraries(     chemtensor_test PRIVATE ${BLAS_LIBRARIES} ${LAPACKE_LIBRARIES} ${HDF5_LIBRARIES})
