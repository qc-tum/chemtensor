/// \file linked_list.c
/// \brief Doubly linked list.

#include <assert.h>
#include "linked_list.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Append the item pointed to by 'd' to the end of the linked list.
///
void linked_list_append(struct linked_list* list, void* d)
{
	struct linked_list_node* node = aligned_alloc(MEM_DATA_ALIGN, sizeof(struct linked_list_node));
	node->data = d;
	node->next = NULL;

	if (list->size == 0)
	{
		node->prev = NULL;

		list->head = node;
		list->tail = node;
	}
	else
	{
		node->prev = list->tail;

		list->tail->next = node;
		list->tail = node;
	}

	list->size++;
}


//________________________________________________________________________________________________________________________
///
/// \brief Prepend the item pointed to by 'd' to the beginning of the linked list.
///
void linked_list_prepend(struct linked_list* list, void* d)
{
	struct linked_list_node* node = aligned_alloc(MEM_DATA_ALIGN, sizeof(struct linked_list_node));
	node->data = d;
	node->prev = NULL;

	if (list->size == 0)
	{
		node->next = NULL;

		list->head = node;
		list->tail = node;
	}
	else
	{
		node->next = list->head;

		list->head->prev = node;
		list->head = node;
	}

	list->size++;
}


//________________________________________________________________________________________________________________________
///
/// \brief Insert the item pointed to by 'd' after 'node' into the linked list.
///
void linked_list_insert_after_node(struct linked_list* list, struct linked_list_node* node, void* d)
{
	assert(list->size > 0);

	struct linked_list_node* new_node = aligned_alloc(MEM_DATA_ALIGN, sizeof(struct linked_list_node));
	new_node->data = d;
	new_node->prev = node;
	new_node->next = node->next;

	if (node->next == NULL) {
		assert(node == list->tail);
		list->tail = new_node;
	}
	else {
		node->next->prev = new_node;
	}

	node->next = new_node;

	list->size++;
}


//________________________________________________________________________________________________________________________
///
/// \brief Insert the item pointed to by 'd' before 'node' into the linked list.
///
void linked_list_insert_before_node(struct linked_list* list, struct linked_list_node* node, void* d)
{
	assert(list->size > 0);

	struct linked_list_node* new_node = aligned_alloc(MEM_DATA_ALIGN, sizeof(struct linked_list_node));
	new_node->data = d;
	new_node->next = node;
	new_node->prev = node->prev;

	if (node->prev == NULL) {
		assert(node == list->head);
		list->head = new_node;
	}
	else {
		node->prev->next = new_node;
	}

	node->prev = new_node;

	list->size++;
}


//________________________________________________________________________________________________________________________
///
/// \brief Remove 'node' from the linked list (assuming that it is actually part of the linked list), free its memory and
/// return data pointer stored in the node.
///
void* linked_list_remove_node(struct linked_list* list, struct linked_list_node* node)
{
	assert(list->size > 0);

	if (node->prev == NULL)
	{
		assert(node == list->head);
		list->head = node->next;
	}
	else
	{
		node->prev->next = node->next;
	}

	if (node->next == NULL)
	{
		assert(node == list->tail);
		list->tail = node->prev;
	}
	else
	{
		node->next->prev = node->prev;
	}

	list->size--;

	// free memory of node (but not its data)
	void* data = node->data;
	aligned_free(node);

	return data;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a linked list (free memory).
///
/// The function 'free_func' is called for each data pointer.
///
void delete_linked_list(struct linked_list* list, void (*free_func)(void*))
{
	struct linked_list_node* node = list->head;
	while (node != NULL)
	{
		free_func(node->data);
		struct linked_list_node* next = node->next;
		aligned_free(node);
		node = next;
	}

	list->head = NULL;
	list->tail = NULL;
	list->size = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Internal consistency check of the doubly linked list structure.
///
bool linked_list_is_consistent(const struct linked_list* list)
{
	if (list->size == 0)
	{
		return (list->head == NULL) && (list->tail == NULL);
	}

	if (list->head == NULL) {
		return false;
	}
	if (list->tail == NULL) {
		return false;
	}

	if (list->head->prev != NULL) {
		return false;
	}
	if (list->tail->next != NULL) {
		return false;
	}

	long count = 1;
	const struct linked_list_node* node = list->head;
	while (node->next != NULL)
	{
		if (node->next->prev != node) {
			return false;
		}
		node = node->next;
		count++;
	}
	if (node != list->tail) {
		return false;
	}
	if (count != list->size) {
		return false;
	}

	return true;
}
