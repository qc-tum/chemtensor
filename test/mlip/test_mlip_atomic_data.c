#include "mlip_atomic_data.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_mlip_atomic_data_from_configuration()
{
	// dimethyl carbonate molecule
	const double positions[] = {
		-0.06354920,   0.02615914,   0.03273994,
		-1.09745824,  -0.56171083,  -0.33384407,
		 0.59339279,   0.04178514,   1.16893291,
		 0.70731479,   0.58054012,  -0.92796504,
		-0.18480919,  -0.38314587,   2.27627182,
		 0.11790480,   0.31317315,  -2.18905210,
		 0.50081283,  -0.52887988,   3.16355896,
		-0.99130619,   0.37122315,   2.57026887,
		-0.79898220,  -1.19689691,   1.93909299,
		 0.82173383,   0.39293513,  -3.04121518,
		-0.51975018,  -0.63449883,  -2.23208618,
		-0.68493521,   1.15580308,  -2.35394812,
	};
	qnumber atomic_numbers[] = {
		6, 8, 8, 8, 6, 6, 1, 1, 1, 1, 1, 1,
	};

	// single or double precision data types
	for (int idtype = 0; idtype <= CT_DOUBLE_REAL; idtype++)
	{
		struct atomic_configuration config;
		const ct_long dim_pos[2] = { 12, 3 };
		allocate_dense_tensor(idtype, 2, dim_pos, &config.positions);
		if (config.positions.dtype == CT_SINGLE_REAL)
		{
			float* pdata = config.positions.data;
			for (int j = 0; j < (int)ARRLEN(positions); j++) {
				pdata[j] = (float)positions[j];
			}
		}
		else
		{
			memcpy(config.positions.data, positions, sizeof(positions));
		}
		config.atomic_numbers = atomic_numbers;

		// model parameters
		qnumber z_table[] = { 1, 6, 8 };
		// ensure that 'r_cut' is large enough to store any real numeric type
		double r_cut;
		numeric_from_double(3.0, idtype, &r_cut);
		const struct mlip_model_parameters params = {
			.z_table = z_table,
			.num_species = ARRLEN(z_table),
			.r_cut = &r_cut,
			.dtype = idtype,
		};

		struct mlip_atomic_data atomic_data;
		mlip_atomic_data_from_configuration(&config, &params, &atomic_data);

		// reference data
		struct dense_tensor node_attrs_ref;
		{
			const ct_long dim_attrs[2] = { 12, 3 };
			allocate_dense_tensor(idtype, 2, dim_attrs, &node_attrs_ref);
			// one-hot encoding
			const double node_attrs_ref_data[] = {
				0., 1., 0.,
				0., 0., 1.,
				0., 0., 1.,
				0., 0., 1.,
				0., 1., 0.,
				0., 1., 0.,
				1., 0., 0.,
				1., 0., 0.,
				1., 0., 0.,
				1., 0., 0.,
				1., 0., 0.,
				1., 0., 0.,
			};
			if (node_attrs_ref.dtype == CT_SINGLE_REAL)
			{
				float* adata = node_attrs_ref.data;
				for (int j = 0; j < (int)ARRLEN(node_attrs_ref_data); j++) {
					adata[j] = (float)node_attrs_ref_data[j];
				}
			}
			else
			{
				memcpy(node_attrs_ref.data, node_attrs_ref_data, sizeof(node_attrs_ref_data));
			}
		}
		// neighbor map
		int neigh_n_0[] = { 1,  2,  3,  4,  5,  7,  8, 10, 11 };
		int neigh_n_1[] = { 0,  2,  3,  4,  5,  8, 10, 11 };
		int neigh_n_2[] = { 0,  1,  3,  4,  6,  7,  8 };
		int neigh_n_3[] = { 0,  1,  2,  5,  9, 10, 11 };
		int neigh_n_4[] = { 0,  1,  2,  6,  7,  8 };
		int neigh_n_5[] = { 0,  1,  3,  9, 10, 11 };
		int neigh_n_6[] = { 2,  4,  7,  8 };
		int neigh_n_7[] = { 0,  2,  4,  6,  8 };
		int neigh_n_8[] = { 0,  1,  2,  4,  6,  7 };
		int neigh_n_9[] = { 3,  5, 10, 11 };
		int neigh_n10[] = { 0,  1,  3,  5,  9, 11 };
		int neigh_n11[] = { 0,  1,  3,  5,  9, 10 };
		int* neighbor_map_ref[] = {
			neigh_n_0, neigh_n_1, neigh_n_2, neigh_n_3,
			neigh_n_4, neigh_n_5, neigh_n_6, neigh_n_7,
			neigh_n_8, neigh_n_9, neigh_n10, neigh_n11,
		};
		int num_neighbors_ref[] = {
			ARRLEN(neigh_n_0), ARRLEN(neigh_n_1), ARRLEN(neigh_n_2), ARRLEN(neigh_n_3),
			ARRLEN(neigh_n_4), ARRLEN(neigh_n_5), ARRLEN(neigh_n_6), ARRLEN(neigh_n_7),
			ARRLEN(neigh_n_8), ARRLEN(neigh_n_9), ARRLEN(neigh_n10), ARRLEN(neigh_n11),
		};
		struct abstract_graph neighbors_ref = {
			.neighbor_map  = neighbor_map_ref,
			.num_neighbors = num_neighbors_ref,
			.num_nodes = ARRLEN(neighbor_map_ref),
		};

		// compare
		if (!dense_tensor_allclose(&atomic_data.positions, &config.positions, 0.)) {
			return "atomic positions do not match initial configuration";
		}
		if (!dense_tensor_allclose(&atomic_data.node_attrs, &node_attrs_ref, 0.)) {
			return "node attributes do not match reference tensor";
		}
		if (!abstract_graph_equal(&atomic_data.neighbors, &neighbors_ref)) {
			return "node neighbor graph does not match reference";
		}

		delete_dense_tensor(&node_attrs_ref);
		delete_mlip_atomic_data(&atomic_data);
		delete_dense_tensor(&config.positions);
	}

	return 0;
}
