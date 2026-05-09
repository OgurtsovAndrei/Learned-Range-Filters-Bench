#include <iostream>
#include <vector>
#include <chrono>
#include <sux/bits/SimpleSelectZeroHalf.hpp>

int main() {
    size_t n = 16 * 1024 * 1024;
    std::vector<uint64_t> bits((n + 63) / 64, 0xFFFFFFFFFFFFFFFFULL); // all ones
    
    // Add a zero at the end
    size_t zero_pos = n - 1;
    bits[zero_pos / 64] &= ~(1ULL << (zero_pos % 64));
    
    // Add a zero at the beginning
    bits[0] &= ~1ULL;
    
    sux::bits::SimpleSelectZeroHalf<> sz(bits.data(), n);
    
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1000; ++i) {
        volatile uint64_t p = sz.selectZero(1); // Find the zero at the end
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    std::cout << "1000 selectZero(1) calls on 16M-bit gap took: " << diff.count() << "s" << std::endl;
    std::cout << "Avg time per call: " << (diff.count() / 1000.0) * 1e6 << " us" << std::endl;
    
    return 0;
}
