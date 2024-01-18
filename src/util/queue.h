/// \file queue.h
/// \brief Queue data structure.

#pragma once

#include <stdbool.h>


//________________________________________________________________________________________________________________________
///
/// \brief Node of a queue data structure.
///
struct queue_node
{
	void* data;               //!< pointer to data entry
	struct queue_node* next;  //!< pointer to next node
};


//________________________________________________________________________________________________________________________
///
/// \brief Queue data structure.
///
struct queue
{
	struct queue_node* head;  //!< pointer to head node
	struct queue_node* tail;  //!< pointer to tail node
};

void enqueue(struct queue* q, void* d);
void* dequeue(struct queue* q);

void* peek_queue(const struct queue* q);

bool queue_is_empty(const struct queue* q);

void free_queue(struct queue* q, void (*free_func)(void*));
