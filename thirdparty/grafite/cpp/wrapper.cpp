#include "../wrapper.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <grafite/grafite.hpp>

// Use the default RangeEmptinessDS (ef_sux_vector when SUCCINCT_LIB_SUX is defined)
using GrafiteDefaultFilter = grafite::filter<>;

// Grafite's constructor throws std::runtime_error when r > max(S), i.e. when
// bpk > log2(u/n) + 2. In that regime the hash function degenerates to a
// constant shift and a lossless Elias-Fano encoding of the raw keys is
// strictly better than any approximate filter. Upstream uses this as a hard
// "use the right tool" assertion, but for benchmarking parity with the
// thesis' own SodaARE+ERE backend (which silently switches to a lossless
// encoding via the same path) we want the wrapper to behave the same way:
// when this happens, build an Elias-Fano directly over the sorted raw keys
// and let queries hit it without any hashing. False positive rate becomes 0
// deterministically; space cost is the same as `ef_sux_vector(keys)`.
//
// We do NOT modify upstream grafite.hpp. We catch the throw at the wrapper
// level and route through a sibling field.
struct GrafiteHandle {
    GrafiteDefaultFilter*    approx   = nullptr;  // hashed-mode filter
    grafite::ef_sux_vector*  lossless = nullptr;  // raw-key Elias-Fano
    uint64_t                 first    = 0;
    uint64_t                 last     = 0;
    bool                     empty    = true;     // n == 0 short-circuit
};

namespace {

// Build the lossless fallback by sorting + deduplicating the raw keys and
// building an Elias-Fano directly on them. Caller takes ownership.
GrafiteHandle* build_lossless_fallback(std::vector<uint64_t> keys) {
    auto* h = new GrafiteHandle{};
    if (keys.empty()) {
        h->empty = true;
        return h;
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    h->lossless = new grafite::ef_sux_vector(keys.begin(), keys.end(), false);
    h->first    = keys.front();
    h->last     = keys.back();
    h->empty    = false;
    return h;
}

GrafiteHandle* build_approx(std::vector<uint64_t> vec, double bpk) {
    auto* h = new GrafiteHandle{};
    if (vec.empty()) {
        h->empty = true;
        return h;
    }
    try {
        h->approx = new GrafiteDefaultFilter(vec.begin(), vec.end(), bpk);
        h->empty  = false;
    } catch (const std::runtime_error&) {
        delete h;
        return build_lossless_fallback(std::move(vec));
    }
    return h;
}

GrafiteHandle* build_approx_eps_l(std::vector<uint64_t> vec, double eps, uint64_t L) {
    auto* h = new GrafiteHandle{};
    if (vec.empty()) {
        h->empty = true;
        return h;
    }
    try {
        h->approx = new GrafiteDefaultFilter(vec.begin(), vec.end(), eps,
                static_cast<typename std::vector<uint64_t>::value_type>(L));
        h->empty  = false;
    } catch (const std::runtime_error&) {
        delete h;
        return build_lossless_fallback(std::move(vec));
    }
    return h;
}

// Range-emptiness query against the lossless EF.
// Returns true iff any stored key falls in [lo, hi].
bool lossless_query(const GrafiteHandle& h, uint64_t lo, uint64_t hi) {
    if (h.empty || lo > hi) return false;
    if (lo > h.last || hi < h.first) return false;
    // ef_sux_vector::check_presence(a, b) computes rank(a) != rank(b+1).
    // Avoid hi+1 wraparound: if hi == UINT64_MAX, the answer is just
    // (lo <= h.last), which we have already verified above.
    if (hi == static_cast<uint64_t>(-1)) return lo <= h.last;
    return h.lossless->check_presence(lo, hi);
}

} // anonymous namespace

extern "C" {

GrafitePtr grafite_new(const uint64_t* keys, size_t n, double bpk) {
    std::vector<uint64_t> vec(keys, keys + n);
    return static_cast<GrafitePtr>(build_approx(std::move(vec), bpk));
}

GrafitePtr grafite_new_eps_l(const uint64_t* keys, size_t n, double eps, uint64_t L) {
    std::vector<uint64_t> vec(keys, keys + n);
    return static_cast<GrafitePtr>(build_approx_eps_l(std::move(vec), eps, L));
}

int grafite_query(GrafitePtr ptr, uint64_t lo, uint64_t hi) {
    auto* h = static_cast<GrafiteHandle*>(ptr);
    if (h == nullptr || h->empty) return 0;
    if (h->approx) {
        return h->approx->query(lo, hi) ? 1 : 0;
    }
    return lossless_query(*h, lo, hi) ? 1 : 0;
}

void grafite_query_batch(GrafitePtr ptr, const uint64_t* queries, size_t count, uint8_t* results) {
    auto* h = static_cast<GrafiteHandle*>(ptr);
    if (h == nullptr || h->empty) {
        std::memset(results, 0, count);
        return;
    }
    if (h->approx) {
        for (size_t i = 0; i < count; ++i) {
            results[i] = h->approx->query(queries[2 * i], queries[2 * i + 1]) ? 1 : 0;
        }
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        results[i] = lossless_query(*h, queries[2 * i], queries[2 * i + 1]) ? 1 : 0;
    }
}

uint64_t grafite_size_bits(GrafitePtr ptr) {
    auto* h = static_cast<GrafiteHandle*>(ptr);
    if (h == nullptr || h->empty) return 0;
    if (h->approx) {
        return static_cast<uint64_t>(h->approx->size()) * 8;
    }
    return static_cast<uint64_t>(h->lossless->size()) * 8;
}

void grafite_free(GrafitePtr ptr) {
    auto* h = static_cast<GrafiteHandle*>(ptr);
    if (h == nullptr) return;
    delete h->approx;
    delete h->lossless;
    delete h;
}

}
