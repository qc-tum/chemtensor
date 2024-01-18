#include <string.h>
#include "hash_table.h"
#include "config.h"


struct key_struct
{
	int i;
	char *s;
};


//
// Fowler-Noll-Vo FNV-1a (64-bit) hash function, see
// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function and
// http://www.isthe.com/chongo/tech/comp/fnv/index.html
//
static uint64_t fnv1a(const char *key)
{
	uint64_t hash = 14695981039346656037U;
	while (*key)
	{
		hash = (hash ^ (*key)) * 1099511628211U;
		key++;
	}
	return hash;
}


static hash_type hash_func(const void* k)
{
	const struct key_struct* key = k;
	hash_type hash = fnv1a(key->s);
	// include integer variable in key
	hash = (hash ^ (key->i)) * 1099511628211U;
	return hash;
}


static bool key_comp(const void* k1, const void* k2)
{
	const struct key_struct* key1 = k1;
	const struct key_struct* key2 = k2;

	return (key1->i == key2->i) && (strcmp(key1->s, key2->s) == 0);
}


char* test_hash_table()
{
	// create a hash table with an artifically small number of buckets, such that collisions will occur
	struct hash_table ht;
	create_hash_table(hash_func, key_comp, sizeof(struct key_struct), 5, &ht);

	struct key_struct keys[9] = {
		{ .i = 13, .s = "first key"   },
		{ .i = -3, .s = "second key"  },
		{ .i = 71, .s = "third key"   },
		{ .i = 62, .s = "fourth key"  },
		{ .i = 56, .s = "fifth key"   },
		{ .i = 42, .s = "sixth key"   },
		{ .i = -7, .s = "seventh key" },
		{ .i = 34, .s = "eighth key"  },
		{ .i = 87, .s = "ninth key"   },
	};
	const short static_vals[9] = { 7, 4, -5, 64, 25, -1, 73, -2, 23 };
	short* vals[9];
	for (int i = 0; i < 9; i++) {
		vals[i] = aligned_alloc(MEM_DATA_ALIGN, sizeof(short));
		*vals[i] = static_vals[i];
	}

	// insert some key-value pairs
	for (int i = 0; i < 6; i++) {
		if (hash_table_insert(&ht, &keys[i], vals[i]) != NULL) {
			return "inserting a new key into hash table should return NULL";
		}
	}
	if (ht.num_entries != 6) {
		return "incorrect number of entry counter in hash table";
	}

	short* pval = hash_table_get(&ht, &keys[1]);
	if (*pval != static_vals[1]) {
		return "retrieved value from hash table does not match expected value";
	}

	if (hash_table_get(&ht, &keys[8]) != NULL) {
		return "hash table returns value despite non-existing key";
	}

	// remove second key
	pval = hash_table_remove(&ht, &keys[1]);
	if (*pval != static_vals[1]) {
		return "returned value after removing key from hash table does not match expected value";
	}
	aligned_free(pval);
	if (ht.num_entries != 5) {
		return "incorrect number of entry counter in hash table";
	}

	// re-insert third key
	short* val2_2 = aligned_alloc(MEM_DATA_ALIGN, sizeof(short));
	*val2_2 = -6;
	pval = hash_table_insert(&ht, &keys[2], val2_2);
	if (*pval != static_vals[2]) {
		return "re-inserting a key should return previous value";
	}
	aligned_free(pval);
	// ensure that hash table actually stores new value
	pval = hash_table_get(&ht, &keys[2]);
	if (*pval != *val2_2) {
		return "retrieved value from hash table does not match expected value";
	}

	// insert some more key-value pairs
	for (int i = 6; i < 9; i++) {
		if (hash_table_insert(&ht, &keys[i], vals[i]) != NULL) {
			return "inserting a new key into hash table should return NULL";
		}
	}

	pval = hash_table_get(&ht, &keys[7]);
	if (*pval != static_vals[7]) {
		return "retrieved value from hash table does not match expected value";
	}

	delete_hash_table(&ht, aligned_free);

	return 0;
}
