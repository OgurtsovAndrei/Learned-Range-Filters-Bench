package bench

import (
	"Thesis/emptiness/are_soda_hash"
	"Thesis/testutils"
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
)

func TestEREBucketStats_SodaARE(t *testing.T) {
	n := 1 << 20 // 1M keys
	rangeLens := []uint64{16, 256, 4096}
	epsilon := 0.01

	type dataset struct {
		name string
		load func() ([]uint64, error)
	}

	datasets := []dataset{
		{"uniform", func() ([]uint64, error) {
			rng := rand.New(rand.NewSource(42))
			keys := make([]uint64, n)
			seen := make(map[uint64]bool, n)
			for i := 0; i < n; {
				v := rng.Uint64()
				if !seen[v] {
					seen[v] = true
					keys[i] = v
					i++
				}
			}
			sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
			return keys, nil
		}},
		{"clustered", func() ([]uint64, error) {
			rng := rand.New(rand.NewSource(42))
			keys, _ := testutils.GenerateClusterDistribution(n, 8, 0.1, rng)
			return keys, nil
		}},
		{"sosd_fb", func() ([]uint64, error) {
			return bucketLoadSOSD64(bucketSOSDPath("fb_200M_uint64"), n)
		}},
		{"sosd_wiki", func() ([]uint64, error) {
			return bucketLoadSOSD64(bucketSOSDPath("wiki_ts_200M_uint64"), n)
		}},
		{"sosd_osm", func() ([]uint64, error) {
			return bucketLoadSOSD64(bucketSOSDPath("osm_cellids_800M_uint64"), n)
		}},
		{"sosd_books", func() ([]uint64, error) {
			return bucketLoadSOSD32(bucketSOSDPath("books_200M_uint32"), n)
		}},
	}

	for _, ds := range datasets {
		keys, err := ds.load()
		if err != nil {
			t.Logf("skip %s: %v", ds.name, err)
			continue
		}

		for _, L := range rangeLens {
			name := fmt.Sprintf("%s/L=%d", ds.name, L)
			t.Run(name, func(t *testing.T) {
				are, err := are_soda_hash.NewSodaARE(keys, L, epsilon)
				if err != nil {
					t.Fatalf("build failed: %v", err)
				}
				stats := are.EREStats()
				fmt.Printf("%-20s L=%-5d | blocks=%-8d non-empty=%-8d avg=%.2f max=%d\n",
					ds.name, L, stats.NumBlocks, stats.NonEmptyBlocks,
					stats.AvgKeysPerBlock, stats.MaxKeysInBlock)
			})
		}
	}
}

func bucketSOSDPath(name string) string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), "sosd_data", name)
}

func bucketLoadSOSD64(path string, maxKeys int) ([]uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var count uint64
	if err := binary.Read(f, binary.LittleEndian, &count); err != nil {
		return nil, err
	}
	readN := int(count)
	if maxKeys > 0 && maxKeys < readN {
		readN = maxKeys
	}
	keys := make([]uint64, readN)
	if err := binary.Read(f, binary.LittleEndian, keys); err != nil {
		return nil, err
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	j := 0
	for i := 1; i < len(keys); i++ {
		if keys[i] != keys[j] {
			j++
			keys[j] = keys[i]
		}
	}
	return keys[:j+1], nil
}

func bucketLoadSOSD32(path string, maxKeys int) ([]uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var count uint64
	if err := binary.Read(f, binary.LittleEndian, &count); err != nil {
		return nil, err
	}
	readN := int(count)
	if maxKeys > 0 && maxKeys < readN {
		readN = maxKeys
	}
	raw := make([]uint32, readN)
	if err := binary.Read(f, binary.LittleEndian, raw); err != nil {
		return nil, err
	}
	keys := make([]uint64, readN)
	for i, v := range raw {
		keys[i] = uint64(v)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	j := 0
	for i := 1; i < len(keys); i++ {
		if keys[i] != keys[j] {
			j++
			keys[j] = keys[i]
		}
	}
	return keys[:j+1], nil
}
