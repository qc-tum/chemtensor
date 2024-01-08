#include "queue.h"
#include "config.h"


char* test_queue()
{
	// empty queue
	struct queue q = { 0 };

	if (!queue_is_empty(&q)) {
		return "expecting empty queue";
	}

	enqueue(&q, (void*)-1);
	if (queue_is_empty(&q)) {
		return "expecting non-empty queue";
	}
	enqueue(&q, (void*)3);
	enqueue(&q, (void*)4);
	if ((int)dequeue(&q) != -1) {
		return "dequeued item does not have expected value";
	}
	if ((int)peek_queue(&q) != 3) {
		return "peeked item of queue does not have expected value";
	}
	enqueue(&q, (void*)7);
	if ((int)dequeue(&q) != 3) {
		return "dequeued item does not have expected value";
	}

	while (q.head != NULL) {
		dequeue(&q);
	}
	if (!queue_is_empty(&q)) {
		return "expecting empty queue";
	}
	enqueue(&q, (void*)5);
	enqueue(&q, (void*)-2);
	if ((int)peek_queue(&q) != 5) {
		return "peeked item of queue does not have expected value";
	}

	free_queue(&q);

	return 0;
}
