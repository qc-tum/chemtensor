
cmake_minimum_required(VERSION 3.15)

project(chemtensor LANGUAGES C)

include(CTest)

find_package(BLAS REQUIRED)
if (NOT BLAS_INCLUDE_DIRS)
    find_path(BLAS_INCLUDE_DIRS cblas.h "/usr/include" "/usr/include/cblas" "/usr/local/include" "/usr/local/include/cblas")
endif()
find_package(LAPACK REQUIRED)
find_path(LAPACK_INCLUDE_DIRS lapack.h "/usr/include" "/usr/include/lapack" "/usr/include/lapacke" "/usr/local/include" "/usr/local/include/lapack" "/usr/local/include/lapacke")
if (LAPACK_INCLUDE_DIRS)
    add_compile_definitions(LAPACK_H_AVAILABLE)
    message(STATUS "Found lapack.h in ${LAPACK_INCLUDE_DIRS}")
else()
    set(LAPACK_INCLUDE_DIRS "")
    message(STATUS "lapack.h not found")
endif()
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(OpenMP)

if (APPLE)
    add_compile_definitions(ACCELERATE_NEW_LAPACK)
endif()

set(CHEMTENSOR_DIRS "src" "src/tensor" "src/state" "src/operator" "src/algorithm" "src/util")
set(CHEMTENSOR_SOURCES "src/tensor/dense_tensor.c" "src/tensor/block_sparse_tensor.c" "src/tensor/qnumber.c" "src/tensor/clebsch_gordan.c" "src/tensor/su2_recoupling.c" "src/tensor/su2_tree.c" "src/tensor/su2_graph.c" "src/tensor/su2_irrep_lists.c" "src/tensor/su2_tensor.c" "src/tensor/su2_util.c" "src/state/mps.c" "src/state/ttns.c" "src/operator/op_chain.c" "src/operator/local_op.c" "src/operator/mpo_graph.c" "src/operator/mpo.c" "src/operator/ttno_graph.c" "src/operator/ttno.c" "src/operator/hamiltonian.c" "src/algorithm/bond_ops.c" "src/algorithm/chain_ops.c" "src/algorithm/tree_ops.c" "src/algorithm/dmrg.c" "src/algorithm/gradient.c" "src/algorithm/bug.c" "src/algorithm/thc.c" "src/util/util.c" "src/util/queue.c" "src/util/linked_list.c" "src/util/hash_table.c" "src/util/abstract_graph.c" "src/util/bipartite_graph.c" "src/util/integer_linear_algebra.c" "src/util/krylov.c" "src/util/runge_kutta.c" "src/util/pcg_basic.c" "src/util/rng.c" "src/util/timing.c")
set(TEST_SOURCES "test/tensor/test_dense_tensor.c" "test/tensor/test_block_sparse_tensor.c" "test/tensor/test_clebsch_gordan.c" "test/tensor/test_su2_tree.c" "test/tensor/test_su2_graph.c" "test/tensor/test_su2_tensor.c" "test/state/test_mps.c" "test/state/test_ttns.c" "test/operator/test_mpo_graph.c" "test/operator/test_mpo.c" "test/operator/test_ttno_graph.c" "test/operator/test_ttno.c" "test/operator/test_hamiltonian.c" "test/algorithm/test_bond_ops.c" "test/algorithm/test_chain_ops.c" "test/algorithm/test_tree_ops.c" "test/algorithm/test_dmrg.c" "test/algorithm/numerical_gradient.c" "test/algorithm/test_gradient.c" "test/algorithm/test_bug.c" "test/algorithm/test_thc.c" "test/util/test_queue.c" "test/util/test_linked_list.c" "test/util/test_hash_table.c" "test/util/test_bipartite_graph.c" "test/util/test_integer_linear_algebra.c" "test/util/test_krylov.c" "test/util/test_runge_kutta.c" "test/run_tests.c")

add_executable(            chemtensor_test ${CHEMTENSOR_SOURCES} ${TEST_SOURCES})
target_include_directories(chemtensor_test PRIVATE ${CHEMTENSOR_DIRS} ${BLAS_INCLUDE_DIRS} ${LAPACK_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
target_link_libraries(     chemtensor_test PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${HDF5_LIBRARIES})
if (OpenMP_C_FOUND)
    target_link_libraries( chemtensor_test PRIVATE OpenMP::OpenMP_C)
endif()

add_library(               chemtensor_pymodule SHARED ${CHEMTENSOR_SOURCES} "python/pymodule.c")
target_include_directories(chemtensor_pymodule PRIVATE ${CHEMTENSOR_DIRS} ${BLAS_INCLUDE_DIRS} ${LAPACK_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(     chemtensor_pymodule PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${HDF5_LIBRARIES} ${Python3_LIBRARIES} Python3::NumPy)

if (OpenMP_C_FOUND)
    target_link_libraries( chemtensor_pymodule PRIVATE OpenMP::OpenMP_C)
endif()
set_target_properties(     chemtensor_pymodule PROPERTIES PREFIX "" OUTPUT_NAME "chemtensor_pymodule" LINKER_LANGUAGE C)

add_executable(            basic_dmrg_fermi_hubbard ${CHEMTENSOR_SOURCES} "examples/dmrg/basic_dmrg_fermi_hubbard.c")
target_include_directories(basic_dmrg_fermi_hubbard PRIVATE ${CHEMTENSOR_DIRS} ${BLAS_INCLUDE_DIRS} ${LAPACK_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
target_link_libraries(     basic_dmrg_fermi_hubbard PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${HDF5_LIBRARIES})
if (OpenMP_C_FOUND)
    target_link_libraries( basic_dmrg_fermi_hubbard PRIVATE OpenMP::OpenMP_C)
endif()

add_executable(            perf_dmrg ${CHEMTENSOR_SOURCES} "perf/perf_dmrg.c")
target_include_directories(perf_dmrg PRIVATE ${CHEMTENSOR_DIRS} ${BLAS_INCLUDE_DIRS} ${LAPACK_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
target_link_libraries(     perf_dmrg PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${HDF5_LIBRARIES})
if (OpenMP_C_FOUND)
    target_link_libraries( perf_dmrg PRIVATE OpenMP::OpenMP_C)
endif()

add_test(NAME chemtensor_test COMMAND chemtensor_test)

install(TARGETS chemtensor_pymodule DESTINATION .)
