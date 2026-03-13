#include "../wrapper.h"

#include <vector>
#include <string>
#include <cstdint>

#include "surf.hpp"

static std::string uint64_to_be_string(uint64_t v) {
    char buf[8];
    buf[0] = (char)((v >> 56) & 0xFF);
    buf[1] = (char)((v >> 48) & 0xFF);
    buf[2] = (char)((v >> 40) & 0xFF);
    buf[3] = (char)((v >> 32) & 0xFF);
    buf[4] = (char)((v >> 24) & 0xFF);
    buf[5] = (char)((v >> 16) & 0xFF);
    buf[6] = (char)((v >>  8) & 0xFF);
    buf[7] = (char)((v      ) & 0xFF);
    return std::string(buf, 8);
}

extern "C" {

SuRFPtr surf_new(const uint64_t* keys, size_t n, int suffix_type, int hash_bits, int real_bits) {
    std::vector<std::string> svec;
    svec.reserve(n);
    for (size_t i = 0; i < n; i++) {
        svec.push_back(uint64_to_be_string(keys[i]));
    }

    surf::SuffixType st;
    switch (suffix_type) {
        case 1:  st = surf::kHash; break;
        case 2:  st = surf::kReal; break;
        default: st = surf::kNone; break;
    }

    auto* f = new surf::SuRF(svec, st,
                              static_cast<uint32_t>(hash_bits),
                              static_cast<uint32_t>(real_bits));
    return static_cast<SuRFPtr>(f);
}

int surf_query(SuRFPtr ptr, uint64_t lo, uint64_t hi) {
    auto* f = static_cast<surf::SuRF*>(ptr);
    std::string lo_s = uint64_to_be_string(lo);
    std::string hi_s = uint64_to_be_string(hi);
    return f->lookupRange(lo_s, true, hi_s, true) ? 1 : 0;
}

uint64_t surf_size_bits(SuRFPtr ptr) {
    auto* f = static_cast<surf::SuRF*>(ptr);
    return static_cast<uint64_t>(f->getMemoryUsage()) * 8;
}

void surf_free(SuRFPtr ptr) {
    auto* f = static_cast<surf::SuRF*>(ptr);
    delete f;
}

}
