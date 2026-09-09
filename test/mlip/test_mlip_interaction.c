#include "mlip_interaction.h"
#include "mlip_embedding.h"
#include "aligned_memory.h"
#include "hdf5_util.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_mlip_interaction_first_layer()
{
	hid_t file = H5Fopen("../test/mlip/data/test_mlip_interaction_first_layer.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_mlip_interaction_first_layer failed";
	}

	// model parameters
	struct mlip_model_parameters params = {
		.num_bessel_func = 8,
		.l_max = 2,
		.num_channels = 8,
		.num_layers = 1,  // use a single layer for this test
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
	// average number of neighbors
	{
		if (read_hdf5_attribute(file, "avg_num_neighbors", H5T_NATIVE_DOUBLE, params.avg_num_neighbors) < 0) {
			return "reading 'avg_num_neighbors' attribute from disk failed";
		}
	}

	struct mlip_model model = {
		.params = params,
		.layers = ct_calloc(params.num_layers, sizeof(struct mlip_model_layer))
	};

	// weight matrix for linear transformation of node features
	{
		hsize_t dim_hsize[2];
		if (get_hdf5_dataset_dims(file, "node_linear_weight", dim_hsize) < 0) {
			return "obtaining dimensions of node transformation weight matrix failed";
		}
		const ct_long dim[2] = { dim_hsize[0], dim_hsize[1] };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &model.layers[0].node_linear);
		// read values from disk
		if (read_hdf5_dataset(file, "node_linear_weight", H5T_NATIVE_DOUBLE, model.layers[0].node_linear.data) < 0) {
			return "reading node transformation weight matrix from disk failed";
		}
	}

	// inner radial MLP layers
	{
		int radial_mlp_nlayers;
		if (read_hdf5_attribute(file, "radial_mlp_nlayers", H5T_NATIVE_INT, &radial_mlp_nlayers) < 0) {
			return "reading 'radial_mlp_nlayers' attribute from disk failed";
		}
		allocate_empty_fully_connected_net_spec(radial_mlp_nlayers, &model.layers[0].radial_mlp);

		double silu_prefactor;
		if (read_hdf5_attribute(file, "silu_prefactor", H5T_NATIVE_DOUBLE, &silu_prefactor) < 0) {
			return "reading 'silu_prefactor' attribute from disk failed";
		}

		for (int i = 0; i < model.layers[0].radial_mlp.nlayers; i++)
		{
			// weight matrix
			{
				char varname[1024];
				sprintf(varname, "radial_mlp_weigth%i", i);
				hsize_t dim_hsize[2];
				if (get_hdf5_dataset_dims(file, varname, dim_hsize) < 0) {
					return "obtaining dimensions of radial MLP weight matrix failed";
				}
				const ct_long dim[2] = { dim_hsize[0], dim_hsize[1] };
				allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &model.layers[0].radial_mlp.layers[i].weight);
				// read values from disk
				if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, model.layers[0].radial_mlp.layers[i].weight.data) < 0) {
					return "reading radial MLP weight matrix from disk failed";
				}
			}

			model.layers[0].radial_mlp.layers[i].normalize = true;
			if (i < model.layers[0].radial_mlp.nlayers - 1)
			{
				model.layers[0].radial_mlp.layers[i].activation_func = ACTIVATION_SILU;
				numeric_from_double(silu_prefactor, params.dtype, model.layers[0].radial_mlp.layers[i].activation_prefactor);
			}
			else
			{
				model.layers[0].radial_mlp.layers[i].activation_func = ACTIVATION_IDENTITY;
				// prefactor not used
			}
		}
	}

	// linear transformation to compute the 'A' functions
	{
		const int ndim = 3;

		// construct the fuse and split tree
		//
		//    1    2
		//     ╲  ╱
		//      ╲╱
		//      │
		//      │
		//      0
		//
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j0f = { .i_ax = 0, .c = { &j1,  &j2  } };
		struct su2_tree_node j0s = { .i_ax = 0, .c = { NULL, NULL } };

		struct su2_fuse_split_tree tree = { .tree_fuse = &j0f, .tree_split = &j0s, .ndim = ndim };

		if (!su2_fuse_split_tree_is_consistent(&tree)) {
			return "internal consistency check for the fuse and split tree failed";
		}

		// outer (logical and auxiliary) 'j' quantum numbers
		qnumber j0list[] = { 0, 2, 4 };
		qnumber j1list[] = { 0, 2, 4 };
		qnumber j2list[] = { 0 };  // auxiliary
		const struct su2_irreducible_list outer_irreps[3] = {
			{ .jlist = j0list, .num = ARRLEN(j0list) },
			{ .jlist = j1list, .num = ARRLEN(j1list) },
			{ .jlist = j2list, .num = ARRLEN(j2list) },
		};

		// degeneracy dimensions, indexed by 'j' quantum numbers
		//                          j:             0              1              2              3              4
		const ct_long dim_degen0[] = { model.params.num_channels, 0, model.params.num_channels, 0, model.params.num_channels };
		const ct_long dim_degen1[] = { model.params.num_channels, 0, model.params.num_channels, 0, model.params.num_channels };
		const ct_long* dim_degen[] = {
			dim_degen0,
			dim_degen1,
		};

		const int ndim_logical   = 2;
		const int ndim_auxiliary = 1;

		struct su2_tensor* atomic_base_linear = &model.layers[0].atomic_base_linear;
		allocate_su2_tensor(CT_DOUBLE_REAL, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, atomic_base_linear);
		assert(atomic_base_linear->charge_sectors.nsec == 3);

		for (int c = 0; c < atomic_base_linear->charge_sectors.nsec; c++)
		{
			const char* symm_label[3] = { "0e", "1o", "2e" };
			char varname[1024];
			sprintf(varname, "atomic_base_linear_weight_%s", symm_label[c]);
			hsize_t dim_hsize[2];
			if (get_hdf5_dataset_dims(file, varname, dim_hsize) < 0) {
				return "obtaining dimensions of atomic base linear transformation weight block failed";
			}
			if (((int)dim_hsize[0] != model.params.num_channels) || ((int)dim_hsize[1] != model.params.num_channels)) {
				return "atomic base linear transformation weight block has incompatible dimension";
			}
			// read values from disk
			if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, atomic_base_linear->degensors[c]->data) < 0) {
				return "reading atomic base linear transformation weight block from disk failed";
			}
		}

		if (!su2_tensor_is_consistent(atomic_base_linear)) {
			return "internal consistency check for SU(2) tensor failed";
		}
	}

	if (!mlip_model_is_consistent(&model)) {
		return "internal consistency check of MLIP model failed";
	}

	// input data
	struct mlip_embedding embedding;
	// node embedding data
	{
		hsize_t dim_hsize[2];
		if (get_hdf5_dataset_dims(file, "node_embedding", dim_hsize) < 0) {
			return "obtaining dimensions of node embedding matrix failed";
		}
		const ct_long dim[2] = { dim_hsize[0], dim_hsize[1] };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &embedding.node);
		// read values from disk
		if (read_hdf5_dataset(file, "node_embedding", H5T_NATIVE_DOUBLE, embedding.node.data) < 0) {
			return "reading matrix entries from disk failed";
		}
	}
	// radial embedding data
	{
		hsize_t dim_hsize[2];
		if (get_hdf5_dataset_dims(file, "radial_embedding", dim_hsize) < 0) {
			return "obtaining dimensions of radial embedding matrix failed";
		}
		const ct_long dim[2] = { dim_hsize[0], dim_hsize[1] };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &embedding.radial);
		// read values from disk
		if (read_hdf5_dataset(file, "radial_embedding", H5T_NATIVE_DOUBLE, embedding.radial.data) < 0) {
			return "reading matrix entries from disk failed";
		}
	}
	// angular embedding data
	{
		hsize_t dim_hsize[2];
		if (get_hdf5_dataset_dims(file, "angular_embedding", dim_hsize) < 0) {
			return "obtaining dimensions of angular embedding matrix failed";
		}
		const ct_long dim[2] = { dim_hsize[0], dim_hsize[1] };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &embedding.angular);
		// read values from disk
		if (read_hdf5_dataset(file, "angular_embedding", H5T_NATIVE_DOUBLE, embedding.angular.data) < 0) {
			return "reading matrix entries from disk failed";
		}
	}
	// edge indices
	{
		hsize_t dim_hsize[2];
		if (get_hdf5_dataset_dims(file, "edge_indices", dim_hsize) < 0) {
			return "obtaining dimensions of edge indices failed";
		}
		if (dim_hsize[1] != 2) {
			return "expecting dimension 2 for second dimension of 'edge_indices'";
		}
		embedding.edge_indices = ct_malloc(dim_hsize[0] * 2 * sizeof(int));
		// read values from disk
		if (read_hdf5_dataset(file, "edge_indices", H5T_NATIVE_INT, embedding.edge_indices) < 0) {
			return "reading matrix entries from disk failed";
		}
	}

	struct dense_tensor atomic_base;
	mlip_interaction_first_layer(&embedding, &model, &atomic_base);

	// reference
	struct dense_tensor atomic_base_ref;
	{
		hsize_t dim_hsize[2];
		if (get_hdf5_dataset_dims(file, "atomic_base", dim_hsize) < 0) {
			return "obtaining dimensions of atomic base tensor failed";
		}
		const ct_long dim[2] = { dim_hsize[0], dim_hsize[1] };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &atomic_base_ref);
		// read values from disk
		if (read_hdf5_dataset(file, "atomic_base", H5T_NATIVE_DOUBLE, atomic_base_ref.data) < 0) {
			return "reading atomic base tensor from disk failed";
		}
	}

	// compare
	if (!dense_tensor_allclose(&atomic_base, &atomic_base_ref, 1e-13)) {
		return "MLIP atomic base tensor resulting from first interaction layer does not match reference";
	}

	delete_dense_tensor(&atomic_base_ref);
	delete_dense_tensor(&atomic_base);
	delete_mlip_embedding(&embedding);
	delete_mlip_model(&model);

	H5Fclose(file);

	return 0;
}
