#include "queue.h"
#include "aligned_memory.h"


char* test_queue()
{
	// empty queue
	struct queue q = { 0 };

	if (!queue_is_empty(&q)) {
		return "expecting empty queue";
	}

	int static_data[6] = { -1, 3, 4, 7, 5, -2 };
	int* data[6];
	for (int i = 0; i < 6; i++) {
		data[i] = aligned_alloc(MEM_DATA_ALIGN, sizeof(int));
		*data[i] = static_data[i];
	}

	enqueue(&q, data[0]);
	if (queue_is_empty(&q)) {
		return "expecting non-empty queue";
	}
	enqueue(&q, data[1]);
	enqueue(&q, data[2]);

	int* d = dequeue(&q);
	if (*d != static_data[0]) {
		return "dequeued item does not have expected value";
	}
	aligned_free(d);

	d = peek_queue(&q);
	if (*d != static_data[1]) {
		return "peeked item of queue does not have expected value";
	}

	enqueue(&q, data[3]);

	d = dequeue(&q);
	if (*d != static_data[1]) {
		return "dequeued item does not have expected value";
	}
	aligned_free(d);

	while (q.head != NULL) {
		aligned_free(dequeue(&q));
	}
	if (!queue_is_empty(&q)) {
		return "expecting empty queue";
	}

	enqueue(&q, data[4]);
	enqueue(&q, data[5]);
	d = peek_queue(&q);
	if (*d != static_data[4]) {
		return "peeked item of queue does not have expected value";
	}

	free_queue(&q, aligned_free);

	return 0;
}
