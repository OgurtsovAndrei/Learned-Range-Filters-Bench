# On-The-Fly Building of Light ZFT Structures

## Problem Statement

To build light (compact) structures like HZFT and AZFT, the original implementation first builds a **heavy ZFastTrie**
which stores:

- All strings in memory as `bits.BitString` extents
- Full tree structure with `Node[V]` pointers
- `Handle2NodeMap` mapping all handles to nodes

For n keys of average length L bits, this requires **O(n × L)** memory just for the strings, plus tree overhead.

**Goal**: Reduce memory usage during construction by building handles on-the-fly from sorted keys.

## Solution Overview

### HZFT: Fully Streaming Construction

The Hollow Z-Fast Trie only needs to store `extentLen` per node (a small integer). This makes true streaming
construction possible.

**Algorithm** (`NewHZFastTrieFromIteratorStreaming`):

1. Process sorted keys one at a time
2. Maintain a stack of "open" internal nodes (depths where branching occurred)
3. For each new key:
    - Compute LCP with previous key
    - "Close" nodes whose extent ends before LCP by emitting their handles
    - Push new branching point to stack if needed
4. After last key, close remaining nodes
5. Build MPH from collected handles

**Key insight**: We only need to keep:

- Previous key (for LCP computation)
- Stack of depths (O(log L) entries)
- Map of handle → extentLen (handles are short prefixes, not full keys)

### AZFT: Reduced-Overhead Construction

Currently, AZFT uses a "reduced-overhead" approach. It still builds a temporary heavy ZFT, but discards it immediately
after construction.

AZFT requires more complex per-node data:
- `extentLen` - length of extent
- `PSig` - hash signature of extent (requires full extent content)
- `parent` - first left-ancestor index
- `minChild`, `minGreaterChild` - leftmost nodes in subtrees
- `rightChild` - right child index
- `Rank` - delimiter index

These relationships depend on tree structure, making truly streaming construction complex.

### AZFT: Two-Pass Light Build (Planned)

To achieve true streaming for AZFT (similar to HZFT), a two-pass approach is used to avoid materializing the full
ZFastTrie.

**Algorithm**:

1. **Pass 1 (Topology)**: Process sorted keys using a stack-based approach to build a lightweight tree of handles and
   basic parent/child pointers. This determines the trie structure without storing full keys.
2. **MPH Construction**: Build the Minimal Perfect Hash from the collected handles.
3. **Pass 2 (Resolution)**: Traverse the lightweight tree to resolve complex attributes (`minChild`, `minGreaterChild`,
   `parent` indices) and populate the final compact `NodeData` array.

This approach reduces peak memory from $O(n \times L)$ to $O(n \times \log L)$.

## Complexity Analysis

### Heavy ZFT Construction (Original)

| Metric            | Complexity         | Notes                                             |
|-------------------|--------------------|---------------------------------------------------|
| Time              | O(n × L × log L)   | n insertions, each O(L × log L) for handle lookup |
| Memory            | O(n × L)           | Stores all n strings of average length L          |
| Memory (detailed) | O(n × L + n × ptr) | Strings + node pointers + Handle2NodeMap          |

### HZFT Streaming Construction (New)

| Metric           | Complexity   | Notes                                      |
|------------------|--------------|--------------------------------------------|
| Time             | O(n × L)     | Single pass, LCP computation O(L) per pair |
| Memory (working) | O(log L)     | Stack depth bounded by max key length      |
| Memory (output)  | O(n × log L) | Handles are O(log L) bits on average       |

**Theoretical memory savings (analytical estimate)**:
From $O(n \times L)$ to $O(n \times \log L)$ for handle storage.

For a theoretical set of $n$ keys of $L$-bit length, this represents an order-of-magnitude reduction in peak memory required for string/handle storage during construction (e.g., from $L$ bits per key to $\approx \log L$ bits per key).

### AZFT Streaming Construction (New)

| Metric         | Complexity                          | Notes                               |
|----------------|-------------------------------------|-------------------------------------|
| Time           | O(n × L × log L)                    | Same as original (still builds ZFT) |
| Memory (peak)  | O(n × L)                            | During ZFT construction             |
| Memory (final) | O(n × log L + n × sizeof(NodeData)) | After ZFT is discarded              |

**Memory savings**:

- No `originalNode` pointers retained in final structure
- ZFT is garbage-collected after construction
- Trie debug reference is nil

## API

### HZFT

```go
// Streaming construction - recommended for production
func NewHZFastTrieFromIteratorStreaming[E UNumber](
iter bits.BitStringIterator,
) (*HZFastTrie[E], error)

// Original heavy construction - kept for compatibility
func NewHZFastTrieFromIterator[E UNumber](
iter bits.BitStringIterator,
) (*HZFastTrie[E], error)
```

### AZFT

```go
// Reduced-overhead construction - recommended for production
func NewApproxZFastTrieFromIteratorStreaming[E, S, I UNumber](
iter bits.BitStringIterator,
saveOriginalTrie bool, // must be false for memory savings
seed uint64,
) (*ApproxZFastTrie[E, S, I], error)

// Original heavy construction - use when debug info needed
func NewApproxZFastTrieWithSeedFromIterator[E, S, I UNumber](
iter bits.BitStringIterator,
saveOriginalTrie bool,
seed uint64,
) (*ApproxZFastTrie[E, S, I], error)
```

## Requirements

Both streaming builders require **sorted input**:

- Use `bits.NewCheckedSortedIterator(iter)` to validate
- Sorting ensures LCP between consecutive keys determines tree structure
- Violation causes error return

## Implementation Details

### HZFT Handle Emission

For each node, we emit:

1. **Descriptor** (handle): prefix at 2-fattest number in (nameLen-1, extentLen]
2. **Pseudo-descriptors**: prefixes at all 2-fattest numbers in (nameLen-1, handleLen)

```go
func (b *streamingBuilder[E]) emit(depth, parentDepth int, key bits.BitString) {
a := uint64(parentDepth)
extentLen := uint64(depth)

// Main descriptor
original := bits.TwoFattest(a, extentLen)
desc := key.Prefix(int(original))
b.kv[desc] = HNodeData[E]{extentLen: E(extentLen)}

// Pseudo-descriptors (map to infinity)
b_pseudo := original - 1
for a < b_pseudo {
ftst := bits.TwoFattest(a, b_pseudo)
b.kv[key.Prefix(int(ftst))] = HNodeData[E]{extentLen: ^E(0)}
b_pseudo = ftst - 1
}
}
```

### Stack-Based Tree Traversal

The stack tracks branching points in the path from root:

```go
type stackItem struct {
depth int // Extent length at this branching point
}

// When LCP decreases, we "close" nodes
for d > lcp {
topDepth := stack[len(stack)-1].depth
p_d := max(lcp, topDepth)

emit(d, p_d, prevKey) // Emit node at depth d

d = p_d
if topDepth == d {
stack = stack[:len(stack)-1] // Pop
}
}
```

## Testing

Tests verify streaming builders produce identical results to heavy builders:

```go
func TestStreamingBuilder_MatchesHeavyBuilder(t *testing.T) {
// Generate random sorted keys
keys := zft.GenerateRandomBitStrings(numKeys, bitLen, r)

// Build both ways
heavyHZFT, _ := NewHZFastTrieFromIterator[uint32](iter)
streamingHZFT, _ := NewHZFastTrieFromIteratorStreaming[uint32](iter)

// Compare on all prefixes
for _, key := range keys {
for prefixLen := 1; prefixLen <= key.Size(); prefixLen++ {
prefix := key.Prefix(prefixLen)
assert(heavy.GetExistingPrefix(prefix) == streaming.GetExistingPrefix(prefix))
}
}
}
```

## Future Work

1. **True streaming AZFT**: Implement stack-based construction that computes parent/child relationships without full ZFT
2. **External memory**: For very large datasets, spill handles to disk during construction
3. **Parallel construction**: Split key ranges and merge handle maps

## Files

- `trie/hzft/hzft_builder.go` - Streaming HZFT builder
- `trie/hzft/hzft_builder_test.go` - Tests
- `trie/azft/azft_builder.go` - Reduced-overhead AZFT builder
- `trie/azft/azft_builder_test.go` - Tests
