package keygen

import (
	"fmt"
	"math/rand"
	"os"
)

// LoadKeysForSpec is a helper for benchmarks to load keys based on distribution name.
func LoadKeysForSpec(distName string, n int) ([]uint64, error) {
	switch distName {
	case "sosd_fb":
		return LoadSOSDUint64(SOSDPath("fb_200M_uint64"), n)
	case "uniform":
		rng := rand.New(rand.NewSource(0xBEEF))
		ks := make([]uint64, n)
		for i := range ks {
			ks[i] = rng.Uint64()
		}
		return ks, nil
	default:
		return nil, nil
	}
}

// CacheOrGenerate tries to load keys from a cache file. If not found, calls generate(),
// saves to cache, and returns the keys. Cache path: {syntheticDataDir}/{distName}_{n}.bin.
func CacheOrGenerate(distName string, n int, generate func() []uint64) []uint64 {
	dir := SyntheticDataPath("")
	if err := os.MkdirAll(dir, 0755); err != nil {
		fmt.Printf("[GEN KEYS] %s n=%d (mkdir failed: %v, generating...)\n", distName, n, err)
		return generate()
	}
	path := fmt.Sprintf("%s/%s_%d.bin", dir, distName, n)
	if keys, err := LoadSOSDUint64(path, 0); err == nil {
		fmt.Printf("[CACHED KEYS] %s n=%d (loaded from %s)\n", distName, n, path)
		return keys
	}
	keys := generate()
	if err := SaveSyntheticKeys(path, keys); err != nil {
		fmt.Printf("[GEN KEYS] %s n=%d (save failed: %v)\n", distName, n, err)
	} else {
		fmt.Printf("[GEN KEYS] %s n=%d (saved to %s)\n", distName, n, path)
	}
	return keys
}
