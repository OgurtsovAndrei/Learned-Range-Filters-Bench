package bench_test

import (
	"Thesis-bench-industry/bench/internal/keygen"
	"math/rand"
)


// saveSyntheticKeys saves keys in SOSD binary format: [uint64 count LE][count × uint64 keys LE].
func saveSyntheticKeys(path string, keys []uint64) error {
	return keygen.SaveSyntheticKeys(path, keys)
}

// loadSyntheticKeys loads keys from SOSD binary format. Returns error if file doesn't exist.
func loadSyntheticKeys(path string) ([]uint64, error) {
	return keygen.LoadSOSDUint64(path, 0)
}

func mask60Keys(keys []uint64) []uint64 {
	return keygen.Mask60Keys(keys)
}

// cacheOrGenerate tries to load keys from a cache file. If not found, calls generate(),
// saves to cache, and returns the keys. Cache path: {cacheDir}/{distName}_{n}.bin.
func cacheOrGenerate(cacheDir, distName string, n int, generate func() []uint64) []uint64 {
	return keygen.CacheOrGenerate(distName, n, generate)
}

// clusterMeta is the JSON-serialisable form of []testutils.ClusterInfo.
type clusterMeta = keygen.ClusterMeta

func generateUniformKeys(n int, rng *rand.Rand) []uint64 {
	return keygen.GenerateUniformKeys(n, rng)
}

func generateSpreadKeys(n int) []uint64 {
	return keygen.GenerateSpreadKeys(n)
}

func generateZipfianKeys(n, nPrefixes int, rng *rand.Rand) ([]uint64, []uint64) {
	return keygen.GenerateZipfianKeys(n, nPrefixes, rng)
}

func generateTemporalKeys(n int, rng *rand.Rand) []uint64 {
	return keygen.GenerateTemporalKeys(n, rng)
}
