# Research Report: Unified Succinct Trie Map (Merged-LERLOC)

## 1. Objective & Hypothesis
The experiment aimed to combine the **Exit-Node Locator** (SHZFT) and the **Range Locator** (RLOC) into a single monolithic structure. 

**Hypothesis**: Since both components derive their key sets from the same underlying trie structure, merging them into a single Minimal Perfect Hash (MPH) would:
1. Reduce memory by eliminating redundant indexing of overlapping keys.
2. Increase query speed by replacing two heavy MPH lookups with one.

## 2. Brainstorming & Analysis
A preliminary analysis of key set overlaps (N=32,768, L=64) showed:
- **SHZFT Keys**: 179,298
- **RLOC Keys**: 120,350
- **Intersection**: 68,435 keys (~57% of RLOC keys are also in SHZFT).
- **Potential Saving**: Merging would reduce the total number of indexed keys from ~300k to ~231k (**23% reduction**).

## 3. Implementation Challenges

### A. The Monotonicity Conflict
To support RLOC's leaf-rank intervals, the index **must** be monotonic (MMPH). We used `LeMonHash` for this. However, SHZFT's Fat Binary Search requires **Membership** verification (is this prefix actually in the trie?).
- `BoomPHF` (used in original SHZFT) provides membership info (returns 0 for non-existent keys).
- `LeMonHash` (PGM-based) is a "blind" rank mapper; it always returns a rank, even for keys not in the original set.

### B. The "Zero-Prefix" Stall
Learned indices like `LeMonHash` require strictly monotonic byte-strings. In bit-tries, prefixes like `"0"`, `"00"`, `"000"` all map to the same byte `0x00`, causing stalls in the C++ learning phase.
- **Solution Developed**: We implemented **Injective Monotonic Encoding** (Terminator Bit). By appending a `1` bit to every BitString before bit-reversal, we ensured that Trie-order perfectly matched Byte-lexicographical order.

## 4. Experimental Results

### Memory Footprint (N=32,768, L=64)
- **Separate (Current LERLOC)**: **~58.9 bits/key**
- **Merged (Unified Index)**: **~75.9 bits/key**

**Finding**: Merging actually **increased** memory usage by ~17 bits/key. 
The overhead of managing three separate bitvectors (`shzftBV`, `rlocBV`, `leafBV`) to distinguish node types within a "blind" `LeMonHash` exceeded the savings from key deduplication.

### Correctness & Performance
- **Correctness**: The Fat Binary Search failed with `LeMonHash` because the search "hallucinated" nodes on non-existent paths due to the lack of membership verification.
- **Performance**: Query time regressed from ~360ns to ~1400ns due to the increased complexity of the unified bitvector logic and CGO overhead for every step of the search.

## 5. Architectural Lessons Learned
1. **Specialization Wins**: The separation of `ExitNodeLocator` and `RangeLocator` is mathematically sound. They rely on fundamentally different properties of hash functions:
    - **SHZFT** needs **Fast Membership** (provided by BBHash/BoomPHF).
    - **RLOC** needs **Monotonicity** (provided by PGM/LeMonHash).
2. **Scale Matters**: deduplication of keys only provides a ~23% reduction in key count, which is not enough to offset the cost of extra metadata required to "unify" the logic.
3. **BBHash Efficiency**: On the scale of 100k-500k keys, `boomphf` (BBHash) is extremely hard to beat in bits-per-entry (~3.5 bits) while providing membership info.

## 6. Next Steps
Based on these findings, we abandoned the "Merged" approach and will focus on optimizing the individual components within the stable separate architecture:
1. **SHZFT Delta Compression**: Replace fixed-width bit-packing with **Elias-Fano** or **VByte** encoding.
2. **BitVector Refactoring**: Replace `rsdic` with a specialized `Rank`-only bitvector to strip `Select` overhead.
