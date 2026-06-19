# Approx Z-Fast Trie (AZFT)

`azft` is a compact, probabilistic implementation of a Z-Fast Trie, optimized for space efficiency in Monotone Minimal Perfect Hashing (MMPH) and Range Filter applications.

## Overview

Unlike a standard Z-Fast Trie that uses explicit pointers and full keys, the `ApproxZFastTrie` uses:
- **Minimal Perfect Hashing (MPH)**: To map node handles (prefixes) to indices in a flat array.
- **Signatures (PSig)**: To probabilistically verify prefixes without storing the full keys.
- **Fat Binary Search**: To find the "Exit Node" (longest existing prefix) in $O(\log w)$ time.

This results in a structure that uses only $O(m (\log \log n + \log \log w))$ bits of space, where $m$ is the number of nodes.

## Detailed Documentation

The implementation includes several subtle "hacks" and optimizations to handle its probabilistic nature:

- **[Lower Bound Candidates](azft_lower_bound.md)**: Explains why `LowerBound()` returns 6 candidates and how they are used by MMPH to identify bucket delimiters.
- **[Length-Consistency Filter](collision_filter.md)**: Describes a correctness-preserving check in `GetExistingPrefix` that significantly reduces the false positive rate.
- **[False Positives & False Negatives](fp_fn_analysis.md)**: A theoretical analysis of how collisions in the MPH and signatures can lead to both types of errors, and how the MMPH builder mitigates this using a Las Vegas approach.

## Key Features

- **Generic Implementation**: Supports different bit-widths for extent lengths (`E`), signatures (`S`), and node indices (`I`) to tune space vs. precision.
- **Memory-Efficient Builder**: Includes a builder (`azft_builder.go`) that constructs the trie from sorted iterators with reduced memory overhead (`NewApproxZFastTrieFromIteratorStreaming`). See [On-The-Fly Building](../OnTheFlyBuild.md) for theoretical background and performance analysis.
- **Property-Based Testing**: Extensive tests in `azft_property_test.go` verify the error rates against theoretical bounds.

## References

- **MMPH Paper**: Section 4 (Probabilistic Z-Fast Tries) and Section 5 (Relative Tries).
- **Hollow Z-Fast Trie**: Related work on fast prefix search.
