package bench_test

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"sort"
)

// saveSyntheticKeys saves keys in SOSD binary format: [uint64 count LE][count × uint64 keys LE].
func saveSyntheticKeys(path string, keys []uint64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := binary.Write(f, binary.LittleEndian, uint64(len(keys))); err != nil {
		return err
	}
	return binary.Write(f, binary.LittleEndian, keys)
}

// loadSyntheticKeys loads keys from SOSD binary format. Returns error if file doesn't exist.
func loadSyntheticKeys(path string) ([]uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var count uint64
	if err := binary.Read(f, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("read count: %w", err)
	}
	keys := make([]uint64, count)
	if err := binary.Read(f, binary.LittleEndian, keys); err != nil {
		return nil, fmt.Errorf("read keys: %w", err)
	}
	return keys, nil
}

// cacheOrGenerate tries to load keys from a cache file. If not found, calls generate(),
// saves to cache, and returns the keys. Cache path: {cacheDir}/{distName}_{n}.bin.
func cacheOrGenerate(cacheDir, distName string, n int, generate func() []uint64) []uint64 {
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		fmt.Printf("[GEN KEYS] %s n=%d (mkdir failed: %v, generating...)\n", distName, n, err)
		return generate()
	}
	path := fmt.Sprintf("%s/%s_%d.bin", cacheDir, distName, n)
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
	seen := make(map[uint64]bool, n)
	keys := make([]uint64, 0, n)
	for len(keys) < n {
		k := rng.Uint64() & mask60
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys
}

func generateSpreadKeys(n int) []uint64 {
	step := (uint64(1) << 60) / uint64(n)
	keys := make([]uint64, n)
	for i := 0; i < n; i++ {
		keys[i] = uint64(i) * step
	}
	return keys
}

func generateZipfianKeys(n, nPrefixes int, rng *rand.Rand) ([]uint64, []uint64) {
	prefixes := make([]uint64, nPrefixes)
	for i := range prefixes {
		prefixes[i] = rng.Uint64() & ((1 << 40) - 1)
	}
	sort.Slice(prefixes, func(i, j int) bool { return prefixes[i] < prefixes[j] })

	nTop := nPrefixes / 10
	nHot := n * 80 / 100

	seen := make(map[uint64]bool, n)
	keys := make([]uint64, 0, n)
	for len(keys) < nHot {
		pref := prefixes[rng.Intn(nTop)]
		k := (pref << 20) | (rng.Uint64() & ((1 << 20) - 1))
		k &= mask60
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	for len(keys) < n {
		pref := prefixes[nTop+rng.Intn(nPrefixes-nTop)]
		k := (pref << 20) | (rng.Uint64() & ((1 << 20) - 1))
		k &= mask60
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys, prefixes
}

func generateTemporalKeys(n int, rng *rand.Rand) []uint64 {
	base := uint64(1) << 50
	step := uint64(1000)
	jitter := float64(step) / 4.0

	raw := make([]uint64, 0, n*2)
	pos := base
	for len(raw) < n*3/2 {
		offset := int64(rng.NormFloat64() * jitter)
		k := uint64(int64(pos) + offset)
		k &= mask60
		raw = append(raw, k)
		pos += step
	}

	// TTL gap: remove early 30% of keys, keep ~10% survivors from that region
	gapEnd := len(raw) * 30 / 100
	survivors := make([]uint64, 0, n)
	survivors = append(survivors, raw[gapEnd:]...)
	for i := 0; i < gapEnd; i++ {
		if rng.Float64() < 0.10 {
			survivors = append(survivors, raw[i])
		}
	}

	seen := make(map[uint64]bool, len(survivors))
	keys := make([]uint64, 0, n)
	for _, k := range survivors {
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	if len(keys) > n {
		keys = keys[:n]
	}
	return keys
}
