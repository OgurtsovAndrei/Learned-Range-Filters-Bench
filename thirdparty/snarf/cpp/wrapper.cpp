#include "../wrapper.h"

#include <vector>
#include <cstdint>

// SNARF places its implementation files in include/ as .cpp files meant to be #included directly.
// snarf.cpp already includes snarf_model.cpp and snarf_bitset.cpp internally.
#include "snarf.cpp"

extern "C" {

SNARFPtr snarf_new(const uint64_t* keys, size_t n, double bpk) {
    std::vector<uint64_t> vec(keys, keys + n);
    auto* f = new snarf_updatable_gcs<uint64_t>();
    f->snarf_init(vec, bpk, /*batch_size=*/100);
    return static_cast<SNARFPtr>(f);
}

int snarf_query(SNARFPtr ptr, uint64_t lo, uint64_t hi) {
    auto* f = static_cast<snarf_updatable_gcs<uint64_t>*>(ptr);
    // range_query returns true = MAY be non-empty (false positive possible)
    return f->range_query(lo, hi) ? 1 : 0;
}

void snarf_query_batch(SNARFPtr ptr, const uint64_t* queries, size_t count, uint8_t* results) {
    auto* f = static_cast<snarf_updatable_gcs<uint64_t>*>(ptr);
    for (size_t i = 0; i < count; i++) {
        uint64_t lo = queries[2 * i];
        uint64_t hi = queries[2 * i + 1];
        results[i] = f->range_query(lo, hi) ? 1 : 0;
    }
}

uint64_t snarf_size_bits(SNARFPtr ptr) {
    auto* f = static_cast<snarf_updatable_gcs<uint64_t>*>(ptr);
    // return_size() returns bytes
    return static_cast<uint64_t>(f->return_size()) * 8;
}

void snarf_free(SNARFPtr ptr) {
    auto* f = static_cast<snarf_updatable_gcs<uint64_t>*>(ptr);
    delete f;
}

}
