#include "linked_list.h"
#include "aligned_memory.h"


static int* construct_integer(int v)
{
	int* p = aligned_alloc(MEM_DATA_ALIGN, sizeof(int));
	*p = v;
	return p;
}


static void linked_list_export_integer_array(const struct linked_list* list, int* arr)
{
	const struct linked_list_node* node = list->head;
	long i = 0;
	while (node != NULL)
	{
		arr[i++] = *((int*)node->data);
		node = node->next;
	}
}


static bool integer_array_equal(const long n, const int* arr1, const int* arr2)
{
	for (long i = 0; i < n; i++)
	{
		if (arr1[i] != arr2[i]) {
			return false;
		}
	}

	return true;
}


char* test_linked_list()
{
	// empty linked list
	struct linked_list list = { 0 };

	if (!linked_list_is_empty(&list)) {
		return "expecting empty linked list";
	}
	if (!linked_list_is_consistent(&list)) {
		return "internal linked list consistency check failed";
	}

	linked_list_append( &list, construct_integer( 6));
	linked_list_append( &list, construct_integer( 3));
	linked_list_prepend(&list, construct_integer( 5));
	linked_list_append( &list, construct_integer(-1));
	linked_list_append( &list, construct_integer(11));

	if (!linked_list_is_consistent(&list)) {
		return "internal linked list consistency check failed";
	}

	if (list.size != 5) {
		return "linked list does not have expected size";
	}
	int* arr0 = aligned_alloc(MEM_DATA_ALIGN, list.size * sizeof(int));
	linked_list_export_integer_array(&list, arr0);
	const int arr0_ref[5] = { 5, 6, 3, -1, 11 };
	if (!integer_array_equal(list.size, arr0, arr0_ref)) {
		return "linked list does not contain expected values";
	}
	aligned_free(arr0);

	linked_list_insert_after_node(&list, list.head->next, construct_integer(2));

	if (!linked_list_is_consistent(&list)) {
		return "internal linked list consistency check failed";
	}

	linked_list_insert_before_node(&list, list.tail, construct_integer(4));

	if (!linked_list_is_consistent(&list)) {
		return "internal linked list consistency check failed";
	}

	int* p1 = linked_list_remove_node(&list, list.tail->prev->prev);
	if (*p1 != -1) {
		return "data pointer of removed node does not store expected value";
	}
	aligned_free(p1);
	int* p2 = linked_list_remove_node(&list, list.head);
	if (*p2 != 5) {
		return "data pointer of removed node does not store expected value";
	}
	aligned_free(p2);

	if (!linked_list_is_consistent(&list)) {
		return "internal linked list consistency check failed";
	}

	linked_list_prepend(&list, construct_integer(9));

	if (!linked_list_is_consistent(&list)) {
		return "internal linked list consistency check failed";
	}

	if (list.size != 6) {
		return "linked list does not have expected size";
	}
	int* arr1 = aligned_alloc(MEM_DATA_ALIGN, list.size * sizeof(int));
	linked_list_export_integer_array(&list, arr1);
	const int arr1_ref[6] = { 9, 6, 2, 3, 4, 11 };
	if (!integer_array_equal(list.size, arr1, arr1_ref)) {
		return "linked list does not contain expected values";
	}
	aligned_free(arr1);

	delete_linked_list(&list, aligned_free);

	return 0;
}
