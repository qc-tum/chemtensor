#include "bipartite_graph.h"
#include "aligned_memory.h"


char* test_bipartite_graph_maximum_cardinality_matching()
{
	hid_t file = H5Fopen("../test/data/test_bipartite_graph_maximum_cardinality_matching.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_bipartite_graph_maximum_cardinality_matching failed";
	}

	int num_u;
	if (read_hdf5_attribute(file, "num_u", H5T_NATIVE_INT, &num_u) < 0) {
		return "reading number of bipartite graph 'U' vertices from disk failed";
	}
	int num_v;
	if (read_hdf5_attribute(file, "num_v", H5T_NATIVE_INT, &num_v) < 0) {
		return "reading number of bipartite graph 'V' vertices from disk failed";
	}

	// edge dataset is a matrix of dimension 'nedges' x 2
	hsize_t edge_dims[2];
	if (get_hdf5_dataset_dims(file, "edges", edge_dims)) {
		return "reading bipartite graph edges from disk failed";
	}
	struct bipartite_graph_edge* edges = aligned_alloc(MEM_DATA_ALIGN, edge_dims[0] * sizeof(struct bipartite_graph_edge));
	if (read_hdf5_dataset(file, "edges", H5T_NATIVE_INT, edges) < 0) {
		return "reading bipartite graph edges from disk failed";
	}

	// initialize bipartite graph structure
	struct bipartite_graph graph;
	init_bipartite_graph(num_u, num_v, edges, edge_dims[0], &graph);

	struct bipartite_graph_matching matching;
	bipartite_graph_maximum_cardinality_matching(&graph, &matching);

	// reference solution
	// matching dataset is a matrix of dimension 'cardinality' x 2
	hsize_t matching_ref_dims[2];
	if (get_hdf5_dataset_dims(file, "matching", matching_ref_dims)) {
		return "reading bipartite graph matching from disk failed";
	}
	struct bipartite_graph_matching matching_ref = { .edges = NULL, .nedges = matching_ref_dims[0] };
	matching_ref.edges = aligned_alloc(MEM_DATA_ALIGN, matching_ref.nedges * sizeof(struct bipartite_graph_edge));
	if (read_hdf5_dataset(file, "matching", H5T_NATIVE_INT, matching_ref.edges) < 0) {
		return "reading bipartite graph matching from disk failed";
	}

	// compare matchings
	if (matching.nedges != matching_ref.nedges) {
		return "bipartite graph matching cardinality does not agree with reference";
	}
	for (int i = 0; i < matching.nedges; i++) {
		if (matching.edges[i].u != matching_ref.edges[i].u || matching.edges[i].v != matching_ref.edges[i].v) {
			return "bipartite graph matching does not agree with reference";
		}
	}

	// clean up
	aligned_free(matching.edges);
	aligned_free(matching_ref.edges);
	delete_bipartite_graph(&graph);
	aligned_free(edges);

	H5Fclose(file);

	return 0;
}


char* test_bipartite_graph_minimum_vertex_cover()
{
	hid_t file = H5Fopen("../test/data/test_bipartite_graph_minimum_vertex_cover.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_bipartite_graph_minimum_vertex_cover failed";
	}

	int num_u;
	if (read_hdf5_attribute(file, "num_u", H5T_NATIVE_INT, &num_u) < 0) {
		return "reading number of bipartite graph 'U' vertices from disk failed";
	}
	int num_v;
	if (read_hdf5_attribute(file, "num_v", H5T_NATIVE_INT, &num_v) < 0) {
		return "reading number of bipartite graph 'V' vertices from disk failed";
	}

	// edge dataset is a matrix of dimension 'nedges' x 2
	hsize_t edge_dims[2];
	if (get_hdf5_dataset_dims(file, "edges", edge_dims)) {
		return "reading bipartite graph edges from disk failed";
	}
	struct bipartite_graph_edge* edges = aligned_alloc(MEM_DATA_ALIGN, edge_dims[0] * sizeof(struct bipartite_graph_edge));
	if (read_hdf5_dataset(file, "edges", H5T_NATIVE_INT, edges) < 0) {
		return "reading bipartite graph edges from disk failed";
	}

	// initialize bipartite graph structure
	struct bipartite_graph graph;
	init_bipartite_graph(num_u, num_v, edges, edge_dims[0], &graph);

	// determine minimum vertex cover
	bool* u_cover_indicator = aligned_alloc(MEM_DATA_ALIGN, graph.num_u * sizeof(bool));
	bool* v_cover_indicator = aligned_alloc(MEM_DATA_ALIGN, graph.num_v * sizeof(bool));
	bipartite_graph_minimum_vertex_cover(&graph, u_cover_indicator, v_cover_indicator);

	// convert to vertex lists
	int num_u_cover = 0;
	int* u_cover = aligned_alloc(MEM_DATA_ALIGN, graph.num_u * sizeof(int));
	for (int u = 0; u < graph.num_u; u++) {
		if (u_cover_indicator[u]) {
			u_cover[num_u_cover] = u;
			num_u_cover++;
		}
	}
	int num_v_cover = 0;
	int* v_cover = aligned_alloc(MEM_DATA_ALIGN, graph.num_v * sizeof(int));
	for (int v = 0; v < graph.num_v; v++) {
		if (v_cover_indicator[v]) {
			v_cover[num_v_cover] = v;
			num_v_cover++;
		}
	}

	// reference solution
	hsize_t num_u_cover_ref;
	if (get_hdf5_dataset_dims(file, "u_cover", &num_u_cover_ref)) {
		return "reading bipartite graph 'U' cover vertices from disk failed";
	}
	int* u_cover_ref = aligned_alloc(MEM_DATA_ALIGN, num_u_cover_ref * sizeof(int));
	if (read_hdf5_dataset(file, "u_cover", H5T_NATIVE_INT, u_cover_ref) < 0) {
		return "reading bipartite graph 'U' cover vertices from disk failed";
	}
	hsize_t num_v_cover_ref;
	if (get_hdf5_dataset_dims(file, "v_cover", &num_v_cover_ref)) {
		return "reading bipartite graph 'V' cover vertices from disk failed";
	}
	int* v_cover_ref = aligned_alloc(MEM_DATA_ALIGN, num_v_cover_ref * sizeof(int));
	if (read_hdf5_dataset(file, "v_cover", H5T_NATIVE_INT, v_cover_ref) < 0) {
		return "reading bipartite graph 'V' cover vertices from disk failed";
	}

	// compare
	if ((hsize_t)num_u_cover != num_u_cover_ref) {
		return "number of 'U' cover vertices does not agree with reference";
	}
	if ((hsize_t)num_v_cover != num_v_cover_ref) {
		return "number of 'V' cover vertices does not agree with reference";
	}
	for (int i = 0; i < num_u_cover; i++) {
		if (u_cover[i] != u_cover_ref[i]) {
			return "'U' cover vertices do not agree with reference";
		}
	}
	for (int j = 0; j < num_v_cover; j++) {
		if (v_cover[j] != v_cover_ref[j]) {
			return "'V' cover vertices do not agree with reference";
		}
	}

	// clean up
	aligned_free(v_cover_ref);
	aligned_free(u_cover_ref);
	aligned_free(v_cover);
	aligned_free(u_cover);
	aligned_free(v_cover_indicator);
	aligned_free(u_cover_indicator);
	delete_bipartite_graph(&graph);
	aligned_free(edges);

	H5Fclose(file);

	return 0;
}
