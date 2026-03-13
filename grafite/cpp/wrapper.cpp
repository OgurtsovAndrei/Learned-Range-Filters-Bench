#include "../wrapper.h"

#include <vector>
#include <grafite/grafite.hpp>

// Use the default RangeEmptinessDS (ef_sux_vector when SUCCINCT_LIB_SUX is defined)
using GrafiteDefaultFilter = grafite::filter<>;

extern "C" {

GrafitePtr grafite_new(const uint64_t* keys, size_t n, double bpk) {
    std::vector<uint64_t> vec(keys, keys + n);
    auto* f = new GrafiteDefaultFilter(vec.begin(), vec.end(), bpk);
    return static_cast<GrafitePtr>(f);
}

int grafite_query(GrafitePtr ptr, uint64_t lo, uint64_t hi) {
    auto* f = static_cast<GrafiteDefaultFilter*>(ptr);
    // query returns true if range MAY be non-empty
    return f->query(lo, hi) ? 1 : 0;
}

uint64_t grafite_size_bits(GrafitePtr ptr) {
    auto* f = static_cast<GrafiteDefaultFilter*>(ptr);
    // size() returns bytes
    return static_cast<uint64_t>(f->size()) * 8;
}

void grafite_free(GrafitePtr ptr) {
    auto* f = static_cast<GrafiteDefaultFilter*>(ptr);
    delete f;
}

}
