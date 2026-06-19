#ifndef LEMONHASH_WRAPPER_H
#define LEMONHASH_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* LeMonHashPtr;

// The C interface receives an array of C-strings and their lengths.
LeMonHashPtr lemonhash_vl_new(const char* const* strings, const size_t* lengths, size_t num_strings);

uint64_t lemonhash_vl_query(LeMonHashPtr ptr, const char* key_data, size_t key_len);

void lemonhash_vl_query_batch(LeMonHashPtr ptr, const char* const* keys, const size_t* lengths, size_t num_keys, uint64_t* results);

void lemonhash_vl_query_pair(LeMonHashPtr ptr, const char* k1, size_t l1, const char* k2, size_t l2, uint64_t* r1, uint64_t* r2);

size_t lemonhash_vl_space_bits(LeMonHashPtr ptr);

void lemonhash_vl_free(LeMonHashPtr ptr);

#ifdef __cplusplus
}
#endif

#endif
