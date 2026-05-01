#include "../wrapper.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "../ext/dst.h"

namespace {

// Default Rosetta hyperparameters from the upstream bench_rosetta.cpp:
//   sample_rate = 0.1   -> we let the caller decide nq
//   maxlen      = 64    -> u64 keys
//   cutoff      = 0
//   dfs_diff    = 100
//   bfs_diff    = 32
constexpr size_t kMaxLen      = 64;
constexpr size_t kCutoff      = 0;
constexpr size_t kDfsDiff     = 100;
constexpr size_t kBfsDiff     = 32;

// Wrap DstFilter together with its cached size-in-bytes (so we don't have to
// rebuild the on-disk serialization on every SizeInBits() call).
struct RosettaHandle {
    DstFilter<BloomFilter<>, false>* dst;
    size_t                           size_bytes;
};

// Build the empirical query-length distribution by replaying (left,right)
// sample pairs through a stat-keeping DstFilter seeded with a single zero key.
// This mirrors bench_rosetta.cpp's modelling phase.
std::vector<size_t> ComputeQDist(const uint64_t* sample_left,
                                 const uint64_t* sample_right,
                                 size_t          nq) {
    DstFilter<BloomFilter<true>, true> dst_stat(
        kDfsDiff, kBfsDiff,
        [](std::vector<size_t> x) -> std::vector<size_t> {
            for (size_t i = 0; i < x.size(); ++i) { x[i] *= 1.44; }
            return x;
        });

    std::vector<Bitwise> tmp;
    tmp.emplace_back(false, kMaxLen);
    dst_stat.AddKeys(tmp);

    for (size_t i = 0; i < nq; ++i) {
        (void)dst_stat.Query(Bitwise(sample_left[i]), Bitwise(sample_right[i]));
    }
    return std::vector<size_t>(dst_stat.qdist_.begin(), dst_stat.qdist_.end());
}

}  // namespace

extern "C" {

RosettaPtr rosetta_new(
    const uint64_t* keys, size_t n,
    double bpk,
    const uint64_t* sample_left, const uint64_t* sample_right, size_t nq) {

    if (n == 0) return nullptr;

    // DstFilter::AddKeys expects a sorted, deduplicated key sequence (it walks
    // LCPs between consecutive keys to derive the level distribution).
    std::vector<uint64_t> sorted(keys, keys + n);
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    std::vector<size_t> qdist = ComputeQDist(sample_left, sample_right, nq);

    auto* dst = new DstFilter<BloomFilter<>, false>(
        kDfsDiff, kBfsDiff,
        [bpk, qdist](std::vector<size_t> x) -> std::vector<size_t> {
            return calc_dst(std::move(x), bpk, qdist, kCutoff);
        });

    std::vector<Bitwise> bitwise_keys;
    bitwise_keys.reserve(sorted.size());
    for (uint64_t k : sorted) {
        bitwise_keys.emplace_back(k);
    }
    dst->AddKeys(bitwise_keys);

    auto ser = dst->serialize();
    size_t size_bytes = ser.second;
    delete[] ser.first;

    auto* h = new RosettaHandle{dst, size_bytes};
    return static_cast<RosettaPtr>(h);
}

int rosetta_query(RosettaPtr p, uint64_t lo, uint64_t hi) {
    if (p == nullptr) return 0;
    auto* h = static_cast<RosettaHandle*>(p);
    // Bench contract is closed [lo, hi]; Rosetta's range Query is half-open
    // [from, to). Translate hi -> hi+1, with overflow guard.
    if (lo == hi) {
        return h->dst->Query(Bitwise(lo)) ? 1 : 0;
    }
    if (hi == UINT64_MAX) {
        // Half-open [lo, UINT64_MAX+1) covers full upper tail; emulate by
        // probing the closed range [lo, UINT64_MAX-1] and the singleton
        // UINT64_MAX. (This is a rare path in practice.)
        if (h->dst->Query(Bitwise(lo), Bitwise(hi))) return 1;
        return h->dst->Query(Bitwise(hi)) ? 1 : 0;
    }
    return h->dst->Query(Bitwise(lo), Bitwise(hi + 1)) ? 1 : 0;
}

void rosetta_query_batch(RosettaPtr p, const uint64_t* queries, size_t count, uint8_t* results) {
    for (size_t i = 0; i < count; ++i) {
        results[i] = static_cast<uint8_t>(
            rosetta_query(p, queries[2 * i], queries[2 * i + 1]));
    }
}

uint64_t rosetta_size_bits(RosettaPtr p) {
    if (p == nullptr) return 0;
    auto* h = static_cast<RosettaHandle*>(p);
    return static_cast<uint64_t>(h->size_bytes) * 8;
}

void rosetta_free(RosettaPtr p) {
    if (p == nullptr) return;
    auto* h = static_cast<RosettaHandle*>(p);
    delete h->dst;
    delete h;
}

}  // extern "C"
