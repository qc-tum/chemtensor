#include "mlip_embedding.h"
#include "aligned_memory.h"
#include "hdf5_util.h"


char* test_mlip_embedding_from_atomic_data()
{
	hid_t file = H5Fopen("../test/mlip/data/test_mlip_embedding_from_atomic_data.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_mlip_embedding_from_atomic_data failed";
	}

	// number of nodes, i.e., number of atoms
	ct_long num_nodes;

	struct atomic_configuration config;
	// atomic positions
	{
		hsize_t dim_hsize[2];
		if (get_hdf5_dataset_dims(file, "atomic_positions", dim_hsize) < 0) {
			return "obtaining dimensions of atomic positions matrix failed";
		}
		const ct_long dim[2] = { dim_hsize[0], dim_hsize[1] };
		if (dim[1] != 3) {
			return "unexpected second dimension of atomic position matrix (expecting 3 Cartesian coordinates)";
		}
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &config.positions);
		// read values from disk
		if (read_hdf5_dataset(file, "atomic_positions", H5T_NATIVE_DOUBLE, config.positions.data) < 0) {
			return "reading atomic positions from disk failed";
		}

		num_nodes = dim[0];
	}
	// atomic numbers
	{
		config.atomic_numbers = ct_malloc(num_nodes * sizeof(qnumber));
		if (read_hdf5_dataset(file, "atomic_numbers", H5T_NATIVE_INT, config.atomic_numbers) < 0) {
			return "reading atomic numbers from disk failed";
		}
	}

	// model parameters
	struct mlip_model_parameters params = {
		.num_bessel_func = 8,
		.l_max = 2,
		.num_channels = 8,
		.num_layers = 2,
		.dtype = CT_DOUBLE_REAL,
	};
	// z_table
	{
		hsize_t dim[1];
		if (get_hdf5_attribute_dims(file, "z_table", dim) < 0) {
			return "reading dimensions of 'z_table' attribute from disk failed";
		}
		params.num_species = dim[0];

		params.z_table = ct_malloc(params.num_species * sizeof(qnumber));
		if (read_hdf5_attribute(file, "z_table", H5T_NATIVE_INT, params.z_table) < 0) {
			return "reading 'z_table' attribute from disk failed";
		}
	}
	// cutoff radius
	{
		if (read_hdf5_attribute(file, "r_cut", H5T_NATIVE_DOUBLE, params.r_cut) < 0) {
			return "reading 'r_cut' attribute from disk failed";
		}
	}
	// order of the polynomial cutoff function
	{
		if (read_hdf5_attribute(file, "p_cut", H5T_NATIVE_INT, &params.p_cut) < 0) {
			return "reading 'p_cut' attribute from disk failed";
		}
	}
	// average number of neighbors (unused)
	{
		memcpy(params.avg_num_neighbors, numeric_one(CT_DOUBLE_REAL), sizeof(double));
	}

	if (!mlip_model_parameters_are_consistent(&params)) {
		return "internal consistency check of MLIP model parameters failed";
	}

	struct mlip_model model = {
		.params = params,
	};
	// species embedding weight matrix
	{
		const ct_long dim[2] = { params.num_channels, params.num_species };
		allocate_dense_tensor(params.dtype, 2, dim, &model.species_embedding_weight);
		// read species embedding weight matrix from disk
		if (read_hdf5_dataset(file, "species_embedding_weight", H5T_NATIVE_DOUBLE, model.species_embedding_weight.data) < 0) {
			return "reading tensor entries from disk failed";
		}
	}

	struct mlip_atomic_data atomic_data;
	mlip_atomic_data_from_configuration(&config, &params, &atomic_data);

	struct mlip_embedding embedding;
	mlip_embedding_from_atomic_data(&atomic_data, &model, &embedding);

	// reference node embedding data
	struct dense_tensor node_embedding_ref;
	{
		hsize_t dim_hsize[2];
		if (get_hdf5_dataset_dims(file, "node_embedding", dim_hsize) < 0) {
			return "obtaining dimensions of reference node embedding matrix failed";
		}
		const ct_long dim[2] = { dim_hsize[0], dim_hsize[1] };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &node_embedding_ref);
		// read values from disk
		if (read_hdf5_dataset(file, "node_embedding", H5T_NATIVE_DOUBLE, node_embedding_ref.data) < 0) {
			return "reading matrix entries from disk failed";
		}
	}
	// reference radial embedding data
	struct dense_tensor radial_embedding_ref;
	{
		hsize_t dim_hsize[2];
		if (get_hdf5_dataset_dims(file, "radial_embedding", dim_hsize) < 0) {
			return "obtaining dimensions of reference radial embedding matrix failed";
		}
		const ct_long dim[2] = { dim_hsize[0], dim_hsize[1] };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &radial_embedding_ref);
		// read values from disk
		if (read_hdf5_dataset(file, "radial_embedding", H5T_NATIVE_DOUBLE, radial_embedding_ref.data) < 0) {
			return "reading matrix entries from disk failed";
		}
	}
	// reference angular embedding data
	struct dense_tensor angular_embedding_ref;
	{
		hsize_t dim_hsize[2];
		if (get_hdf5_dataset_dims(file, "angular_embedding", dim_hsize) < 0) {
			return "obtaining dimensions of reference angular embedding matrix failed";
		}
		const ct_long dim[2] = { dim_hsize[0], dim_hsize[1] };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &angular_embedding_ref);
		// read values from disk
		if (read_hdf5_dataset(file, "angular_embedding", H5T_NATIVE_DOUBLE, angular_embedding_ref.data) < 0) {
			return "reading matrix entries from disk failed";
		}
	}

	// compare
	if (!dense_tensor_allclose(&embedding.node, &node_embedding_ref, 1e-13)) {
		return "MLIP node embedding data does not match reference";
	}
	if (!dense_tensor_allclose(&embedding.radial, &radial_embedding_ref, 1e-13)) {
		return "MLIP radial embedding data does not match reference";
	}
	if (!dense_tensor_allclose(&embedding.angular, &angular_embedding_ref, 1e-13)) {
		return "MLIP angular embedding data does not match reference";
	}

	delete_dense_tensor(&angular_embedding_ref);
	delete_dense_tensor(&radial_embedding_ref);
	delete_dense_tensor(&node_embedding_ref);
	delete_mlip_embedding(&embedding);
	delete_mlip_atomic_data(&atomic_data);
	delete_dense_tensor(&model.species_embedding_weight);
	ct_free(params.z_table);
	ct_free(config.atomic_numbers);
	delete_dense_tensor(&config.positions);

	H5Fclose(file);

	return 0;
}
