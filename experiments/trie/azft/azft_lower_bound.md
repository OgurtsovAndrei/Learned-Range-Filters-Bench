# Lower Bound Candidates in Approximate Z-Fast Trie

This note explains the `LowerBound` function in `trie/azft/approx_z_fast_trie.go` and why it returns 6 candidates instead of a single result.

## The Problem: Exit Node vs. Lower Bound

The `GetExistingPrefix` function (Fat Binary Search) identifies the **Exit Node**: the longest prefix of query key `x` that exists as an internal node (extent) in the Z-Fast Trie. 

For **Approximate Range Emptiness** and **Fast Prefix Search**, the exit node is often sufficient. However, for **Monotone Minimal Perfect Hashing (MMPH)** using a **Relative Trie**, we need the **Lower Bound**: the first bucket delimiter `d` in the trie such that `d >= x`.

In a compacted trie (Z-Fast), the exit node `alpha` for key `x` does not directly reveal the lower bound because:
1.  **Z-Fast Compaction:** The trie skip intervals mean `x` might diverge from the trie's paths between nodes.
2.  **Lack of Full Keys:** The approximate trie only stores signatures and extents, not the full keys. Without the full keys, we cannot perform bitwise comparison during the search to decide which branch to take.

## The Solution: A Candidate Shortlist

Instead of trying to find the exact lower bound (which is impossible without full keys), we return a set of **6 candidates** that represent all structural possibilities for where the lower bound might reside relative to the Exit Node.

> **TODO: Detailed documentation with pictures explaining each case is required here.**

### The 6 Candidates

Let `node` be the Exit Node found by `GetExistingPrefix(x)`.

1.  **`node.minChild`**: The lexicographically smallest leaf in the subtree of the exit node. This is the lower bound if `x` is a prefix of some key in this subtree.
2.  **`node.minGreaterChild`**: If `x` "diverges" from the exit node to the left (i.e., `x` is smaller than the smallest key in the exit node's left branch), the lower bound is the smallest key in the first branch to the right.
3.  **`getMinGreaterFromParent` (cand3)**: Traverses up from the exit node to find the first ancestor that has a "greater" child (a branch to the right of the path taken to the exit node).
4.  **`getGreaterFromParent` (cand4)**: Traverses up to find the first right-sibling of an ancestor.
5.  **`node.rightChild`**: The immediate right sibling of the exit node.
6.  **`node` (cand6)**: The exit node itself (useful if the exit node is a leaf).

## Integration with MMPH

The caller (`MonotoneHashWithTrie.GetRank`) receives these 6 candidates and uses them to identify the correct bucket:

```go
for _, candidate := range candidates {
    // 1. Map node to bucket index (candidate.Rank)
    // 2. Compare query key x against bucket.delimiter (Full Key Comparison)
    // 3. The first candidate that satisfies (delimiter >= x) is the lower bound.
```

By returning 6 candidates, we complement the $O(\log w)$ bitwise search with a small constant number of full key comparisons against stored bucket delimiters. This allows us to identify the exact lower bound without storing full keys in the trie, satisfying the "Relative Ranking" requirements of the MMPH paper while maintaining $O(\log w)$ total query time.

## References

- Paper: `papers/MonotoneMinimalPerfectHashing.pdf` (Section 5: A relative trie).
- Implementation: `trie/azft/approx_z_fast_trie.go` (`LowerBound`).
- Context: `mmph/relative_trie/hash.go` (how candidates are used).
