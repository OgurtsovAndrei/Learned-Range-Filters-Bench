# LeMonHash SIGSEGV on Non-Member Queries

## Problem

`LeMonHashVL` (C++ MMPH via CGo) crashes with SIGSEGV when `lemonhash_vl_query` is called with a key that was **not** in the construction set.

```
SIGSEGV: segmentation violation
PC=0x1031a3d98 sigcode=2 addr=0x261404
signal arrived during cgo execution

Thesis/mmph/lemonhash._Cfunc_lemonhash_vl_query(...)
Thesis/locators/lemon_rloc.(*LeMonRangeLocator).Query(...)
Thesis/locators/lemon_lerloc.(*LeMonLocalExactRangeLocator).WeakPrefixSearch(...)
Thesis/emptiness/ere_theoretical.(*TheoreticalExactRangeEmptiness).isRangeEmptyInBlock(...)
```

## Root Cause

LeMonHash is a Monotone Minimal Perfect Hash. It guarantees a valid rank **only** for keys present in the construction set. For non-member keys, behavior is undefined — the C++ code accesses invalid memory.

### Call Chain

1. `isRangeEmptyInBlock(blockIdx, a, b)` calls `WeakPrefixSearch(prefix)` with a prefix derived from the query value
2. `WeakPrefixSearch` -> `shzft.GetExistingPrefix(prefix)` -> finds an exit node in the trie
3. `rl.Query(exitNode)` -> `lh.Rank(exitNode.TrimTrailingZeros())` -> CGo call into C++
4. If `TrimTrailingZeros()` produces a byte representation not matching any key in LeMonHash -> **SIGSEGV**

### Why Pure-Go `rloc.GenericRangeLocator` Is Safe

```go
// rloc.go:260-264
lexRankLeft := rl.mmph.GetRank(xArrowBs)
if lexRankLeft == -1 {
    return 0, 0, fmt.Errorf("key not found in structure")
}
```

The pure-Go MMPH returns `-1` for non-members. `lemon_rloc` does not check — LeMonHash cannot return "not found".

### Why Locator Tests Pass

Tests in `lemon_lerloc/lerloc_test.go` only query with **sub-prefixes of existing keys**:
```go
for _, key := range keys {
    for prefixLen := uint32(0); prefixLen <= key.Size(); prefixLen++ {
        prefix = key.Prefix(int(prefixLen))
        lerl.WeakPrefixSearch(prefix)  // prefix always leads to a valid exit node
    }
}
```

The crash only occurs with arbitrary query prefixes (from `isRangeEmptyInBlock`), where `a` and `b` are not keys from the set.

## Applied Fix

Replaced `lemon_lerloc.LeMonLocalExactRangeLocator` with `lerloc.CompactLocalExactRangeLocator`:
- Uses `SuccinctHZFastTrie` + pure-Go `rloc.GenericRangeLocator`
- Safe for arbitrary queries
- All tests pass, no SIGSEGV

## Potential Fixes for lemon_rloc (If LeMonHash Is Needed)

1. **Bounds check**: validate key byte length before `lh.Rank()` — if `len(data) != expected length`, return a fallback value
2. **Membership filter**: add a Bloom filter or xor filter before LeMonHash to verify membership
3. **C++ fix**: wrap `operator()` in LeMonHashVL with bounds checking in the bucket mapper
