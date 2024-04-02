/// \file linked_list.h
/// \brief Doubly linked list.

#pragma once

#include <stdbool.h>


//________________________________________________________________________________________________________________________
///
/// \brief Doubly linked list node.
///
struct linked_list_node
{
	void* data;                     //!< pointer to data entry
	struct linked_list_node* prev;  //!< pointer to previous node
	struct linked_list_node* next;  //!< pointer to next node
};


//________________________________________________________________________________________________________________________
///
/// \brief Doubly linked list data structure.
///
struct linked_list
{
	struct linked_list_node* head;  //!< pointer to head node
	struct linked_list_node* tail;  //!< pointer to tail node
	long size;                      //!< number of entries
};


//________________________________________________________________________________________________________________________
///
/// \brief Indicate whether the linked list is empty.
///
static inline bool linked_list_is_empty(const struct linked_list* list)
{
	return list->size == 0;
}


void linked_list_append(struct linked_list* list, void* d);

void linked_list_prepend(struct linked_list* list, void* d);


void linked_list_insert_after_node(struct linked_list* list, struct linked_list_node* node, void* d);

void linked_list_insert_before_node(struct linked_list* list, struct linked_list_node* node, void* d);


void* linked_list_remove_node(struct linked_list* list, struct linked_list_node* node);


void delete_linked_list(struct linked_list* list, void (*free_func)(void*));


bool linked_list_is_consistent(const struct linked_list* list);
