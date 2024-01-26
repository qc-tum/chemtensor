/// \file hash_table.c
/// \brief Separate chaining hash table.

#include "hash_table.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Initialize and allocate memory for a hash table
///
void create_hash_table(hash_table_key_comp* key_equal, hash_function_type* hash_func, const size_t key_size, const long num_buckets, struct hash_table* ht)
{
	ht->key_equal   = key_equal;
	ht->hash_func   = hash_func;
	ht->key_size    = key_size;
	ht->buckets     = aligned_calloc(MEM_DATA_ALIGN, num_buckets, sizeof(struct hash_table_entry*));
	ht->num_buckets = num_buckets;
	ht->num_entries = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a hash table (free memory).
///
/// The function 'free_func' is called for each value pointer stored in the hash table.
///
void delete_hash_table(struct hash_table* ht, void (*free_func)(void*))
{
	for (long i = 0; i < ht->num_buckets; i++)
	{
		struct hash_table_entry* entry = ht->buckets[i];
		while (entry != NULL)
		{
			aligned_free(entry->key);
			free_func(entry->val);
			struct hash_table_entry* next = entry->next;
			aligned_free(entry);
			entry = next;
		}
	}
	aligned_free(ht->buckets);

	ht->num_buckets = 0;
	ht->num_entries = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Insert an entry into the hash table; if key already exists,
/// replace previous value and return pointer to old value, otherwise return NULL.
///
void* hash_table_insert(struct hash_table* ht, void* key, void* val)
{
	// compute bucket index of 'key'
	const long i = ht->hash_func(key) % ht->num_buckets;

	// p_entry is whatever was pointing at entry
	struct hash_table_entry** p_entry = &ht->buckets[i];
	for (; (*p_entry) != NULL; p_entry = &(*p_entry)->next)
	{
		// current entry
		struct hash_table_entry* entry = *p_entry;

		if (ht->key_equal(key, entry->key))
		{
			// key already exists...
			void* ret = entry->val;
			entry->val = val;
			return ret;
		}
	}

	// key not found, insert entry
	struct hash_table_entry* entry = aligned_alloc(MEM_DATA_ALIGN, sizeof(struct hash_table_entry));
	// copy key
	entry->key = aligned_alloc(MEM_DATA_ALIGN, ht->key_size);
	memcpy(entry->key, key, ht->key_size);
	entry->val = val;
	entry->next = NULL;
	// set pointer in linked list
	*p_entry = entry;
	ht->num_entries++;

	return NULL;
}


//________________________________________________________________________________________________________________________
///
/// \brief Return a pointer to the value corresponding to 'key'; if the key is not found, return NULL.
///
void* hash_table_get(const struct hash_table* ht, void* key)
{
	// compute bucket index of 'key'
	const long i = ht->hash_func(key) % ht->num_buckets;

	struct hash_table_entry* entry;
	for (entry = ht->buckets[i]; entry != NULL; entry = entry->next)
	{
		if (ht->key_equal(key, entry->key)) {
			return entry->val;
		}
	}

	// not found
	return NULL;
}


//________________________________________________________________________________________________________________________
///
/// \brief Remove entry with given key from hash table and return corresponding value; if the key cannot be found, return NULL.
///
void* hash_table_remove(struct hash_table* ht, void* key)
{
	// compute bucket index of 'key'
	const long i = ht->hash_func(key) % ht->num_buckets;

	// p_entry is whatever was pointing at current entry
	struct hash_table_entry** p_entry = &ht->buckets[i];
	for (; (*p_entry) != NULL; p_entry = &(*p_entry)->next)
	{
		// current entry
		struct hash_table_entry* entry = *p_entry;

		if (ht->key_equal(key, entry->key))
		{
			// key found

			(*p_entry) = entry->next;   // redirect pointer, effectively removing current entry from linked list
			void* val = entry->val;     // keep reference to current value
			aligned_free(entry->key);   // delete current entry
			aligned_free(entry);
			ht->num_entries--;

			return val;
		}
	}

	// not found
	return NULL;
}
