package bench_test

import (
	"Thesis-bench-industry/bench/internal/keygen"
	"fmt"
	"math/rand"
	"os"
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
	dir := keygen.SyntheticDataPath("")
	if err := os.MkdirAll(dir, 0755); err != nil {
		fmt.Printf("[GEN KEYS] %s n=%d (mkdir failed: %v, generating...)\n", distName, n, err)
		return generate()
	}
	path := fmt.Sprintf("%s/%s_%d.bin", dir, distName, n)
	if keys, err := loadSyntheticKeys(path); err == nil {
		fmt.Printf("[CACHED KEYS] %s n=%d (loaded from %s)\n", distName, n, path)
		return keys
	}
	keys := generate()
	if err := saveSyntheticKeys(path, keys); err != nil {
		fmt.Printf("[GEN KEYS] %s n=%d (save failed: %v)\n", distName, n, err)
	} else {
		fmt.Printf("[GEN KEYS] %s n=%d (saved to %s)\n", distName, n, path)
	}
	return keys
}

// clusterMeta is the JSON-serialisable form of []testutils.ClusterInfo.
type clusterMeta struct {
	Center uint64  `json:"center"`
	Stddev float64 `json:"stddev"`
}

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

