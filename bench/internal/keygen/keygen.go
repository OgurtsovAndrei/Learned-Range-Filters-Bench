package keygen

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

const (
	// Mask60 is used to mask keys for CGo filters that only support 60-bit keys.
	Mask60 = (uint64(1) << 60) - 1

	sosdDataDir      = "sosd_data"
	syntheticDataDir = "synthetic_data"
)

// ClusterMeta is the JSON-serialisable form of cluster info.
type ClusterMeta struct {
	Center uint64  `json:"center"`
	Stddev float64 `json:"stddev"`
}

// datasetSentinelTrim returns the number of tail sentinel keys to drop after
// sort+dedup for datasets known to contain synthetic boundary values at the end
// of the key space (e.g. evenly-spaced uint64-max sentinels in fb_200M, or
// uint32-boundary sentinels in books_200M). These sentinels stretch the key
// range to [0, 2^64) and make density histograms unreadable.
func datasetSentinelTrim(path string) int {
	switch {
	case strings.Contains(path, "fb_200M_uint64"):
		return 21 // 9 evenly-spaced uint64-boundary sentinels + 12 sparse outliers; dense cluster ends at 77308821508
	case strings.Contains(path, "books_200M_uint32"):
		return 9 // 9 evenly-spaced uint32-boundary sentinels; dense cluster ends at 2^30 = 1073741824
	default:
		return 0
	}
}

func SOSDPath(name string) string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), "..", "..", sosdDataDir, name)
}

func SyntheticDataPath(name string) string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), "..", "..", syntheticDataDir, name)
}

func SyntheticFile(dist string, n int) string {
	if n > (1 << 24) {
		return fmt.Sprintf("%s_256M_uint64", dist)
	}
	return fmt.Sprintf("%s_16M_uint64", dist)
}

func SaveSyntheticKeys(path string, keys []uint64) error {
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

func LoadSOSDUint64(path string, maxKeys int) ([]uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var count uint64
	if err := binary.Read(f, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("read count: %w", err)
	}

	readN := int(count)
	if maxKeys > 0 && maxKeys < readN {
		readN = maxKeys
	}

	keys := make([]uint64, readN)
	if err := binary.Read(f, binary.LittleEndian, keys); err != nil {
		return nil, fmt.Errorf("read keys: %w", err)
	}

	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	if len(keys) == 0 {
		return keys, nil
	}
	j := 0
	for i := 1; i < len(keys); i++ {
		if keys[i] != keys[j] {
			j++
			keys[j] = keys[i]
		}
	}
	keys = keys[:j+1]
	if trim := datasetSentinelTrim(path); trim > 0 && len(keys) > trim {
		keys = keys[:len(keys)-trim]
	}
	return keys, nil
}

func LoadSOSDUint32(path string, maxKeys int) ([]uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var count uint64
	if err := binary.Read(f, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("read count: %w", err)
	}

	readN := int(count)
	if maxKeys > 0 && maxKeys < readN {
		readN = maxKeys
	}

	keys32 := make([]uint32, readN)
	if err := binary.Read(f, binary.LittleEndian, keys32); err != nil {
		return nil, fmt.Errorf("read keys: %w", err)
	}

	keys := make([]uint64, len(keys32))
	for i, v := range keys32 {
		keys[i] = uint64(v)
	}

	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	if len(keys) == 0 {
		return keys, nil
	}
	j := 0
	for i := 1; i < len(keys); i++ {
		if keys[i] != keys[j] {
			j++
			keys[j] = keys[i]
		}
	}
	keys = keys[:j+1]
	if trim := datasetSentinelTrim(path); trim > 0 && len(keys) > trim {
		keys = keys[:len(keys)-trim]
	}
	return keys, nil
}

func Mask60Keys(keys []uint64) []uint64 {
	masked := make([]uint64, len(keys))
	for i, k := range keys {
		masked[i] = k & Mask60
	}
	return masked
}

func GenerateUniformKeys(n int, rng *rand.Rand) []uint64 {
	seen := make(map[uint64]bool, n)
	keys := make([]uint64, 0, n)
	for len(keys) < n {
		k := rng.Uint64() & Mask60
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys
}

func GenerateSpreadKeys(n int) []uint64 {
	step := (uint64(1) << 60) / uint64(n)
	keys := make([]uint64, n)
	for i := 0; i < n; i++ {
		keys[i] = uint64(i) * step
	}
	return keys
}

func GenerateZipfianKeys(n, nPrefixes int, rng *rand.Rand) ([]uint64, []uint64) {
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
		k &= Mask60
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	for len(keys) < n {
		pref := prefixes[nTop+rng.Intn(nPrefixes-nTop)]
		k := (pref << 20) | (rng.Uint64() & ((1 << 20) - 1))
		k &= Mask60
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys, prefixes
}

func GenerateTemporalKeys(n int, rng *rand.Rand) []uint64 {
	base := uint64(1) << 50
	step := uint64(1000)
	jitter := float64(step) / 4.0

	raw := make([]uint64, 0, n*2)
	pos := base
	for len(raw) < n*3/2 {
		offset := int64(rng.NormFloat64() * jitter)
		k := uint64(int64(pos) + offset)
		k &= Mask60
		raw = append(raw, k)
		pos += step
	}

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
