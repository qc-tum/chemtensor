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
	hash_function_type* hash_func;      //!< hash function
	hash_table_key_comp* key_eq;        //!< key comparison function
	size_t key_size;                    //!< size of a key, in bytes
	struct hash_table_entry** buckets;  //!< each bucket is a linked list of entries
	long num_buckets;                   //!< number of buckets
	long num_entries;                   //!< number of entries
};


void create_hash_table(hash_function_type* hash_func, hash_table_key_comp* key_eq, const size_t key_size, const long num_buckets, struct hash_table* ht);

void delete_hash_table(struct hash_table* ht, void (*free_func)(void*));


void* hash_table_insert(struct hash_table* ht, void* key, void* val);

void* hash_table_get(const struct hash_table* ht, void* key);

void* hash_table_remove(struct hash_table* ht, void* key);
