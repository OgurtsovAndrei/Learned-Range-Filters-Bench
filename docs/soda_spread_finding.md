# SODA on Spread Distribution: Near-Zero BPK

## Observation

SODA shows BPK ~ 0.0004 on the spread distribution (evenly-spaced keys with step `2^44`),
while on uniform/clustered it behaves normally (BPK 7-17). FPR is also near zero.
The entire filter for 65536 keys occupies 24-40 bits total.

## Root Cause

Spread keys: `key_i = i * 2^44`.

SODA computes hashed keys as:
```
blockIdx = key >> K          // K ~ 20-36 depending on eps and L
ux       = pairwiseHash(blockIdx, hashA, hashB, K)
hx       = (ux + key) & rMask
```

Since `2^44 >> K`, the low K bits of every spread key are **always zero**.
Therefore `hx = ux` for all keys — the key value contributes nothing to the hash.

Furthermore, `blockIdx = i * 2^21` (regular multiples) feeds into pairwiseHash
with limited effective entropy. The result: nearly all 65536 keys hash to the
same 1-2 values in the K-bit universe. `uniqueHashed` collapses, and ERE stores
a trivial 1-node trie.

## Verification

```
eps=0.010 L=   1: K=23  SizeInBits=      27  BPK=0.0004  n=65536
eps=0.010 L=1024: K=33  SizeInBits=      37  BPK=0.0006  n=65536
eps=0.001 L=   1: K=26  SizeInBits=      30  BPK=0.0005  n=65536
```

## Is This a Bug?

No. There are no false negatives — correctness is preserved. SODA legitimately
needs almost no space because the hash function maps all spread keys to the same
point. This is "accidentally perfect" input, not a degradation.

For comparison: Truncation-ARE *degrades* on spread (all prefixes occupied, needs
depth >= log2(n) bits). SODA doesn't degrade — it just gets trivially lucky.

## Implications

- SODA's BPK=0 on spread is not meaningful for comparison with other filters
- A "spread + jitter" variant (adding random low bits) would give fairer results
- This finding is worth mentioning when presenting spread benchmarks
