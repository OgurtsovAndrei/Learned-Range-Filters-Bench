#ifndef GRAFITE_WRAPPER_H
#define GRAFITE_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* GrafitePtr;

GrafitePtr grafite_new(const uint64_t* keys, size_t n, double bpk);
GrafitePtr grafite_new_eps_l(const uint64_t* keys, size_t n, double eps, uint64_t L);
int        grafite_query(GrafitePtr ptr, uint64_t lo, uint64_t hi);
void       grafite_query_batch(GrafitePtr ptr, const uint64_t* queries, size_t count, uint8_t* results);
uint64_t   grafite_size_bits(GrafitePtr ptr);
void       grafite_free(GrafitePtr ptr);

#ifdef __cplusplus
}
#endif

#endif
