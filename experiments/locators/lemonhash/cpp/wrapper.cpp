#include "wrapper.h"

#include <vector>
#include <string>
#include <LeMonHashVL.hpp>

extern "C" {

LeMonHashPtr lemonhash_vl_new(const char* const* strings, const size_t* lengths, size_t num_strings) {
    std::vector<std::string> vec;
    vec.reserve(num_strings);
    for (size_t i = 0; i < num_strings; i++) {
        vec.emplace_back(strings[i], lengths[i]);
    }
    auto* instance = new lemonhash::LeMonHashVL<>(vec);
    return static_cast<LeMonHashPtr>(instance);
}

uint64_t lemonhash_vl_query(LeMonHashPtr ptr, const char* key_data, size_t key_len) {
    auto* instance = static_cast<lemonhash::LeMonHashVL<>*>(ptr);
    return (*instance)(std::string_view(key_data, key_len));
}

void lemonhash_vl_query_batch(LeMonHashPtr ptr, const char* const* keys, const size_t* lengths, size_t num_keys, uint64_t* results) {
    auto* instance = static_cast<lemonhash::LeMonHashVL<>*>(ptr);
    for (size_t i = 0; i < num_keys; ++i) {
        results[i] = (*instance)(std::string_view(keys[i], lengths[i]));
    }
}

void lemonhash_vl_query_pair(LeMonHashPtr ptr, const char* k1, size_t l1, const char* k2, size_t l2, uint64_t* r1, uint64_t* r2) {
    auto* instance = static_cast<lemonhash::LeMonHashVL<>*>(ptr);
    *r1 = (*instance)(std::string_view(k1, l1));
    *r2 = (*instance)(std::string_view(k2, l2));
}

size_t lemonhash_vl_space_bits(LeMonHashPtr ptr) {
    auto* instance = static_cast<lemonhash::LeMonHashVL<>*>(ptr);
    return instance->spaceBits();
}

void lemonhash_vl_free(LeMonHashPtr ptr) {
    auto* instance = static_cast<lemonhash::LeMonHashVL<>*>(ptr);
    delete instance;
}

}
