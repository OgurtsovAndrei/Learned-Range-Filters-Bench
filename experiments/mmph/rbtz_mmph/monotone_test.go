package rbtz

import (
	"crypto/rand"
	"encoding/hex"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTable_Randomized(t *testing.T) {
	t.Parallel()
	sizes := []int{1, 10, 100, 1_000, 10_000, 100_000, 1_000_000}

	for _, size := range sizes {
		keys := buildUniqueKeys(size)

		table := Build(keys)

		for i, key := range keys {
			idx := table.Lookup(key)
			require.Equal(t, uint32(i), idx, "Lookup mismatch for size %d at index %d", size, i)
		}
	}
}

func buildUniqueKeys(size int) []string {
	keys := make([]string, size)
	unique := make(map[string]bool, size)

	for i := 0; i < size; i++ {
		for {
			b := make([]byte, 8)
			_, _ = rand.Read(b)
			s := hex.EncodeToString(b)
			if !unique[s] {
				keys[i] = s
				unique[s] = true
				break
			}
		}
	}
	return keys
}
