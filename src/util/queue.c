/// \file queue.c
/// \brief Queue data structure.

#include <assert.h>
#include "queue.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Enqueue an item.
///
void enqueue(struct queue* q, void* d)
{
	struct queue_node* node = ct_malloc(sizeof(struct queue_node));
	node->data = d;
	node->next = NULL;

	if (q->tail == NULL) {
		q->head = node;
	}
	else {
		q->tail->next = node;
	}
	q->tail = node;
}


//________________________________________________________________________________________________________________________
///
/// \brief Dequeue an item and return a pointer to it.
///
void* dequeue(struct queue* q)
{
	assert(q->head != NULL);
	void* d = q->head->data;
	// move head to next node and delete current head node
	struct queue_node* tmp = q->head;
	q->head = q->head->next;
	ct_free(tmp);
	if (q->head == NULL) {
		q->tail = NULL;
	}
	return d;
}


//________________________________________________________________________________________________________________________
///
/// \brief Return the value of the front item without dequeuing it.
///
void* peek_queue(const struct queue* q)
{
	assert(q->head != NULL);
	return q->head->data;
}


//________________________________________________________________________________________________________________________
///
/// \brief Indicate whether the queue is empty.
///
bool queue_is_empty(const struct queue* q)
{
	return q->head == NULL;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a queue (free memory).
///
/// The function 'free_func' is called for each value pointer stored in the queue.
///
void free_queue(struct queue* q, void (*free_func)(void*))
{
	while (!queue_is_empty(q)) {
		void* d = dequeue(q);
		free_func(d);
	}
}
