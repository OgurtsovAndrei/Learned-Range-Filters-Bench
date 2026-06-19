package rbtz

import (
	"testing"

	"github.com/stretchr/testify/require"
	"golang.org/x/exp/slices"
)

func TestSerializeDeserialize(t *testing.T) {
	t.Parallel()
	keys := []string{
		"alpha", "beta", "gamma", "delta", "epsilon",
		"zeta", "eta", "theta", "iota", "kappa",
	}

	originalTable := Build(keys)

	data, err := originalTable.Serialize()
	require.NoError(t, err)

	require.NotEmpty(t, data)

	var newTable Table
	err = Deserialize(data, &newTable)
	require.NoError(t, err)

	for _, key := range keys {
		expected := originalTable.Lookup(key)
		got := newTable.Lookup(key)
		require.Equal(t, expected, got, "Lookup mismatch for key '%s'", key)
	}

	require.Equal(t, originalTable.level0Mask, newTable.level0Mask, "level0Mask mismatch")
	require.Equal(t, originalTable.level1Mask, newTable.level1Mask, "level1Mask mismatch")
	require.True(t, slices.Equal(originalTable.level0, newTable.level0), "level0 mismatch")
	require.True(t, slices.Equal(originalTable.level1, newTable.level1), "level1 mismatch")
}

func TestDeserialize_InvalidData(t *testing.T) {
	t.Parallel()
	var table Table

	err := Deserialize([]byte{}, &table)
	require.Error(t, err, "expected error for empty data")

	err = Deserialize(make([]byte, 4), &table)
	require.Error(t, err, "expected error for short data")
}
