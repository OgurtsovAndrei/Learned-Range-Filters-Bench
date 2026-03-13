#ifndef SURF_WRAPPER_H
#define SURF_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* SuRFPtr;

// suffix_type: 0=kNone, 1=kHash, 2=kReal
SuRFPtr  surf_new(const uint64_t* keys, size_t n, int suffix_type, int hash_bits, int real_bits);
int      surf_query(SuRFPtr ptr, uint64_t lo, uint64_t hi);
uint64_t surf_size_bits(SuRFPtr ptr);
void     surf_free(SuRFPtr ptr);

#ifdef __cplusplus
}
#endif

#endif
