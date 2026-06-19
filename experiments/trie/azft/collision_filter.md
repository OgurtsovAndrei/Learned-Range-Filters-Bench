# GetExistingPrefix collision filter (length-consistency)

This note explains the small but important check in
`zfasttrie/approx_z_fast_trie.go:GetExistingPrefix`:

```go
if uint64(node.extentLen) < fFast {
	// collision
	b = int32(fFast) - 1
}
```

It is a correctness-preserving filter that removes a class of false positives
created by the MPH-based dictionary and, as a result, reduces the observed
error probability compared to the paper's analysis.

## Context: handles and fat binary search

In the probabilistic z-fast trie (Section 4), each internal node `alpha`
represents a string `p` (the extent) with length `|p|`. Let `q` be the parent
extent. The skip interval of `alpha` is `( |q| .. |p| ]`. The handle length is
the 2-fattest number `f` in that interval, and the dictionary `T` stores:

- key: `p[0:f]`
- value: `(g = |p|, signature(p))`

Therefore, in the *ideal* (paper) model, for any valid handle lookup we must
have:

```
f <= |p|
```

That is, the stored node's extent length is always **at least** the handle
length.

## The practical issue: MPH is not membership

In the implementation, `T` is stored using a minimal perfect hash function
(MPH) over all handles. MPH is not a membership structure, so a query for a
handle that does **not** exist can still return *some* node data.

If this accidental node has extent `p` that is a prefix of the query key
(which is common because we only query keys in `S`), then the signature check
passes **deterministically**:

```
hash(x[0:|p|]) == hash(p)
```

This produces a false positive that is **not** accounted for by the paper's
`2^-S` signature-collision probability (because it is not a hash collision at
all).

## The trick: length-consistency filter

The check `node.extentLen < fFast` rejects a result if the returned node's
extent is **shorter** than the handle length being tested. This is safe because
in a correct lookup we must have `fFast <= extentLen` (see above).

### Why it is correct

Assume a correct handle lookup for length `fFast`. By definition of handles,
`fFast` lies in the skip interval `( |q| .. |p| ]` of the node that should be
returned. Therefore `fFast <= |p|`. Thus any node with `extentLen < fFast` is
**impossible** in the exact model, so rejecting it cannot introduce false
negatives.

### Why it reduces error probability

Many MPH false hits are to nodes with shorter extents (because handles of
shorter length are more common prefixes). For keys in `S`, these shorter
extents are often true prefixes of the key, so the signature check passes with
probability 1. The length-consistency check eliminates this entire class of
false positives, leaving only the "true" signature collisions (`2^-S`) that the
paper's analysis accounts for.

As a result, the empirical success rates are higher than the paper-based
failure bounds, making the theoretical curves look pessimistic in the overlay
plots.

## References

- Paper model: `papers/MMPH/Section-4-Relative-Ranking.md` (probabilistic
  trie, fat binary search) and `papers/MMPH/Section-5-Relative-Trie.md`
  (relative trie on `S`).
- Implementation: `zfasttrie/approx_z_fast_trie.go` (`GetExistingPrefix`).
