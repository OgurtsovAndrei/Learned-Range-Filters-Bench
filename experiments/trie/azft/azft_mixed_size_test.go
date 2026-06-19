package azft

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/trie/zft"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestApproxZFastTrieWithMixedSizeStrings(t *testing.T) {
	t.Parallel()

	// Create a set of mixed-size strings with various scenarios
	keys := []bits.BitString{
		bits.NewFromBinary("1"),     // 1 bit
		bits.NewFromBinary("10"),    // 2 bits
		bits.NewFromBinary("100"),   // 3 bits (10 + trailing zero)
		bits.NewFromBinary("101"),   // 3 bits
		bits.NewFromBinary("1010"),  // 4 bits
		bits.NewFromBinary("10100"), // 5 bits (1010 + trailing zero)
		bits.NewFromBinary("11"),    // 2 bits
		bits.NewFromBinary("110"),   // 3 bits (11 + trailing zero)
		bits.NewFromBinary("1100"),  // 4 bits (11 + trailing zeros)
		bits.NewFromBinary("1101"),  // 4 bits
	}

	// Verify our TrieCompare produces the expected order
	// With TrieCompare: strings with trailing zeros should come before trimmed versions
	expectedOrder := []string{
		"1",     // 1 (single bit)
		"10",    // 10 (trimmed)
		"100",   // 100 (10 + trailing zero)
		"101",   // 101
		"1010",  // 1010
		"10100", // 10100 (1010 + trailing zero)
		"11",    // 11 (trimmed)
		"110",   // 110 (11 + trailing zero)
		"1100",  // 1100 (11 + trailing zeros)
		"1101",  // 1101
	}

	// Sort keys using TrieCompare to verify expected order
	sortedKeys := make([]bits.BitString, len(keys))
	copy(sortedKeys, keys)

	// Manual sort using TrieCompare
	for i := 0; i < len(sortedKeys); i++ {
		for j := i + 1; j < len(sortedKeys); j++ {
			if sortedKeys[i].Compare(sortedKeys[j]) > 0 {
				sortedKeys[i], sortedKeys[j] = sortedKeys[j], sortedKeys[i]
			}
		}
	}

	// Verify the order matches our expectation
	for i, expected := range expectedOrder {
		actual := sortedKeys[i].PrettyString()
		// Extract just the binary part (before the colon)
		for colonIdx := 0; colonIdx < len(actual); colonIdx++ {
			if actual[colonIdx] == ':' {
				actual = actual[:colonIdx]
				break
			}
		}
		require.Equal(t, expected, actual, "Order mismatch at position %d", i)
	}

	// Test Z-fast Trie construction with mixed-size strings
	azft, err := NewApproxZFastTrie[uint16, uint8, uint8](sortedKeys)
	require.NoError(t, err, "failed to build Trie with mixed-size strings")
	require.NotNil(t, azft, "Trie should not be nil")

	// Build reference trie for verification
	referenceTrie, err := zft.BuildFromIterator(bits.NewSliceBitStringIterator(sortedKeys))
	require.NoError(t, err)

	// Verify the Trie can find all keys
	for _, key := range sortedKeys {
		node := azft.GetExistingPrefix(key)
		require.NotNil(t, node, "should find prefix for key %s", key.PrettyString())

		// The node should have an extent that is a prefix of our key
		require.True(t, key.HasPrefix(referenceTrie.GetNode(key.Prefix(int(node.extentLen))).Extent),
			"found extent should be a prefix of key %s", key.PrettyString())
	}

	// Basic functionality test - just verify GetExistingPrefix works
	t.Logf("Trie built successfully with %d nodes", len(azft.data))

	// Test that we can find prefixes (simplified test)
	testKey := bits.NewFromBinary("10")
	node := azft.GetExistingPrefix(testKey)
	if node != nil {
		t.Logf("Found prefix for %s: extent length %d", testKey.PrettyString(), node.extentLen)
	}
}
