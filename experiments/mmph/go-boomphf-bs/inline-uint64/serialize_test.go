package inline_uint64

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func generateRandomKeys(count int) []uint64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	keys := make([]uint64, 0, count)
	keyMap := make(map[uint64]struct{}, count)

	for len(keys) < count {
		newKey := r.Uint64()
		if _, exists := keyMap[newKey]; !exists && newKey != 0 {
			keyMap[newKey] = struct{}{}
			keys = append(keys, newKey)
		}
	}
	return keys
}

func checkDataEquality(t *testing.B, h1, h2 *H, name string) {
	require.Equal(t, len(h1.b), len(h2.b), "%s: h.b length mismatch", name)
	for i := 0; i < len(h1.b); i++ {
		require.Equal(t, h1.b[i], h2.b[i], "%s: h.b[%d] data mismatch", name, i)
	}

	require.Equal(t, len(h1.ranks), len(h2.ranks), "%s: h.ranks length mismatch", name)
	for i := 0; i < len(h1.ranks); i++ {
		require.Equal(t, h1.ranks[i], h2.ranks[i], "%s: h.ranks[%d] data mismatch", name, i)
	}
}

func BenchmarkSerializationIntegrity(b *testing.B) {
	for _, size := range benchKeyCounts {
		keys := generateRandomKeys(size)

		for _, gamma := range testGammas {
			name := fmt.Sprintf("Gamma_%.1f_Size_%d", gamma, size)
			b.Run(name, func(b *testing.B) {
				count := float64(size)

				hOriginal := New(gamma, keys)
				dataSizeInBytes := hOriginal.Size()
				require.NotNil(b, hOriginal, "Failed to create original PHF structure")

				{
					serializedData, err := hOriginal.Serialize()
					require.NoError(b, err, "Serialization failed")

					var hDeserialized H
					err = Deserialize(serializedData, &hDeserialized)
					require.NoError(b, err, "Deserialization failed")
					checkDataEquality(b, hOriginal, &hDeserialized, name)

					b.ReportMetric(float64(dataSizeInBytes)*8/count, "bits/key_in_mem")
					b.ReportMetric(float64(dataSizeInBytes), "bytes_in_mem")
					b.ReportMetric(float64(len(serializedData))*8/count, "bits/key_serialized")
					b.ReportMetric(float64(len(serializedData)), "bytes_serialized")
				}

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					serializedData, err := hOriginal.Serialize()
					require.NoError(b, err, "Serialization failed")
					var hDeserialized H
					err = Deserialize(serializedData, &hDeserialized)
					require.NoError(b, err, "Deserialization failed")
				}
			})
		}
	}
}
