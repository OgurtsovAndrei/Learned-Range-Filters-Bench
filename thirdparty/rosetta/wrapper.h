#ifndef ROSETTA_WRAPPER_H
#define ROSETTA_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* RosettaPtr;

/*
 * Construct a Rosetta filter (DstFilter<BloomFilter<>, false>) over `keys`.
 *
 * `keys` need not be sorted on input; the wrapper sorts internally because
 * DstFilter::AddKeys assumes monotone order (it computes LCPs between
 * consecutive keys).
 *
 * `bpk` is the per-key bit budget. The bit allocation across trie levels
 * is computed by Rosetta's calc_dst() using the empirical query-length
 * distribution derived from (sample_left, sample_right) pairs of length nq.
 * If nq == 0, calc_dst is given an empty qdist (acceptable but degrades
 * Rosetta's BPK shaping).
 */
RosettaPtr rosetta_new(
    const uint64_t* keys, size_t n,
    double bpk,
    const uint64_t* sample_left, const uint64_t* sample_right, size_t nq);

/*
 * Range-emptiness query over CLOSED [lo, hi]. Returns 1 if the range MAY
 * be non-empty (keys present), 0 if definitely empty. Internally translates
 * to Rosetta's half-open [lo, hi+1) interval, with overflow guard.
 */
int        rosetta_query(RosettaPtr p, uint64_t lo, uint64_t hi);

/*
 * Batched closed-range queries. `queries` is laid out as [lo0, hi0, lo1, hi1, ...]
 * of length 2*count. `results[i]` receives 1 (may be non-empty) / 0 (empty).
 */
void       rosetta_query_batch(RosettaPtr p, const uint64_t* queries, size_t count, uint8_t* results);

uint64_t   rosetta_size_bits(RosettaPtr p);
void       rosetta_free(RosettaPtr p);

#ifdef __cplusplus
}
#endif

#endif
