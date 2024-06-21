/// \file hash_table.h
/// \brief Separate chaining hash table.

#pragma once

#include <stdbool.h>
#include <inttypes.h>
#include <memory.h>


//________________________________________________________________________________________________________________________
///
/// \brief Hash table entry, forming a linked list.
///
struct hash_table_entry
{
	struct hash_table_entry* next;  //!< pointer to next entry
	void* key;                      //!< pointer to key
	void* val;                      //!< pointer to corresponding value
};


/// \brief Key equality test function for a hash table.
typedef bool hash_table_key_comp(const void* k1, const void* k2);


/// \brief Hash value data type (unsigned integer to ensure that bucket index is non-negative).
typedef uint64_t hash_type;


/// \brief Hash function type.
typedef hash_type hash_function_type(const void* key);


//________________________________________________________________________________________________________________________
///
/// \brief Associative array using a hash function and linked lists for collisions.
///
struct hash_table
{
	hash_table_key_comp* key_equal;     //!< key equality test function
	hash_function_type* hash_func;      //!< hash function
	size_t key_size;                    //!< size of a key, in bytes
	struct hash_table_entry** buckets;  //!< each bucket is a linked list of entries
	long num_buckets;                   //!< number of buckets
	long num_entries;                   //!< number of entries
};


void create_hash_table(hash_table_key_comp* key_equal, hash_function_type* hash_func, const size_t key_size, const long num_buckets, struct hash_table* ht);

void delete_hash_table(struct hash_table* ht, void (*free_func)(void*));


void* hash_table_insert(struct hash_table* ht, void* key, void* val);

void* hash_table_get(const struct hash_table* ht, void* key);

void* hash_table_remove(struct hash_table* ht, void* key);


//________________________________________________________________________________________________________________________
///
/// \brief Iterator over the entries of a hash table.
///
struct hash_table_iterator
{
	const struct hash_table* table;  //!< reference to hash table
	long i_bucket;                   //!< bucket index
	struct hash_table_entry* entry;  //!< pointer to current entry
};


void init_hash_table_iterator(const struct hash_table* table, struct hash_table_iterator* iter);

bool hash_table_iterator_next(struct hash_table_iterator* iter);

bool hash_table_iterator_is_valid(const struct hash_table_iterator* iter);

const void* hash_table_iterator_get_key(struct hash_table_iterator* iter);
void* hash_table_iterator_get_value(struct hash_table_iterator* iter);
