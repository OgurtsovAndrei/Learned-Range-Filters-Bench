# MMPH Module Notes

Key observations about `relative_trie` with direct links to the detailed sources.

- **Correctness via Las Vegas Validation:** The `ApproxZFastTrie` used for bucket identification is probabilistic and can yield both False Positives and False Negatives (see [FP & FN in AZFT](../../trie/azft/fp_fn_analysis.md)). To ensure 100% correctness for the input key set, the builder:
    1. Builds the trie on bucket delimiters.
    2. Validates the trie by checking every input key resolves to the correct bucket.
    3. Retries with a new seed if validation fails (up to 100 times).
  This converts the Monte Carlo error probability of the trie into a construction-time overhead. See `buildValidatedTrieWithIndices` in `hash.go` and the theoretical analysis in `analysis.md`.
- The bucket size is effectively **fixed to 256**, even though the code comments reference `b = log n`; the `minBucketSize` computed from `log2(n)` is not used. See `relative_trie/hash.go`.
- Correctness relies on **sorted input keys**; the builder does not sort and assumes sorted order for two-pointer validation. See `relative_trie/hash.go` and tests that explicitly sort in `relative_trie/prop_test.go`.
- The MPHF is built on **key hashes**; hash collisions or duplicate keys within a bucket can break rank reconstruction, and there is no explicit uniqueness check. See `relative_trie/hash.go`.
- The trie lookup uses a **length-consistency collision filter** (`extentLen < fFast`) in `GetExistingPrefix`, which removes a class of MPH-induced false positives not modeled in the paper. See `../../trie/azft/collision_filter.md` and `../../trie/azft/azft.go`.
