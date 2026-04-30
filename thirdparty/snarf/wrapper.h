#ifndef SNARF_WRAPPER_H
#define SNARF_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* SNARFPtr;

SNARFPtr snarf_new(const uint64_t* keys, size_t n, double bpk);
int      snarf_query(SNARFPtr ptr, uint64_t lo, uint64_t hi);
void     snarf_query_batch(SNARFPtr ptr, const uint64_t* queries, size_t count, uint8_t* results);
uint64_t snarf_size_bits(SNARFPtr ptr);
void     snarf_free(SNARFPtr ptr);

#ifdef __cplusplus
}
#endif

#endif
