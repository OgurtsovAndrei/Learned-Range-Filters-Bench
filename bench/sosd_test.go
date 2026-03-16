package bench_test

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
)

const sosdDataDir = "sosd_data"

func sosdPath(name string) string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), sosdDataDir, name)
}

// loadSOSDUint64 reads a SOSD binary file: [uint64 count][count × uint64 keys].
// Returns first maxKeys sorted unique keys. If maxKeys <= 0, returns all.
func loadSOSDUint64(path string, maxKeys int) ([]uint64, error) {
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

	// Deduplicate (SOSD files may have rare dupes)
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	j := 0
	for i := 1; i < len(keys); i++ {
		if keys[i] != keys[j] {
			j++
			keys[j] = keys[i]
		}
	}
	keys = keys[:j+1]
	return keys, nil
}

// generateRangeQueries generates uniform queries within the key range.
func generateRangeQueries(keys []uint64, count int, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	minK, maxK := keys[0], keys[len(keys)-1]
	span := maxK - minK
	queries := make([][2]uint64, count)
	for i := range queries {
		a := minK + uint64(rng.Int63n(int64(span)))
		queries[i] = [2]uint64{a, a + rangeLen - 1}
	}
	return queries
}

// Smart query mix weights (fractions must sum to 1.0).
const (
	queryWeightNearKey = 0.50 // query offset from a random key
	queryWeightInGap   = 0.30 // query placed inside a random gap
	queryWeightUniform = 0.20 // uniform random across span
)

// generateSmartQueries generates a mix of query types that follow the data
// distribution. Every returned query is guaranteed to be empty (contains no
// keys from the set). If a candidate query contains a key, it is truncated
// to [a, firstKey-1]. Invalid queries (zero length) are re-rolled.
func generateSmartQueries(keys []uint64, count int, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	n := len(keys)
	minK, maxK := keys[0], keys[n-1]
	span := maxK - minK
	if span == 0 {
		return nil
	}

	nNear := int(float64(count) * queryWeightNearKey)
	nGap := int(float64(count) * queryWeightInGap)
	nUnif := count - nNear - nGap

	// Pre-compute gaps for gap-sampling.
	type gap struct{ lo, hi uint64 }
	gaps := make([]gap, 0, n-1)
	for i := 0; i < n-1; i++ {
		if keys[i+1]-keys[i] > 1 {
			gaps = append(gaps, gap{keys[i] + 1, keys[i+1] - 1})
		}
	}

	queries := make([][2]uint64, 0, count)

	tryAdd := func(a, b uint64) {
		if b < a || a == 0 && b == 0 {
			return
		}
		// Clamp to key range.
		if a < minK && minK > rangeLen {
			a = minK - rangeLen
		}
		if b > maxK+rangeLen {
			b = maxK + rangeLen
		}
		// Find first key >= a.
		idx := sort.Search(n, func(i int) bool { return keys[i] >= a })
		if idx < n && keys[idx] <= b {
			// Query contains key[idx] — truncate to [a, key[idx]-1].
			if keys[idx] == 0 || keys[idx]-1 < a {
				return // can't truncate, skip
			}
			b = keys[idx] - 1
		}
		if b >= a {
			queries = append(queries, [2]uint64{a, b})
		}
	}

	// Near-key queries: pick a random key, offset by [-5*rangeLen, +5*rangeLen].
	for i := 0; i < nNear*2 && len(queries) < nNear; i++ {
		key := keys[rng.Intn(n)]
		offset := rng.Int63n(int64(rangeLen) * 10)
		offset -= int64(rangeLen) * 5
		a := int64(key) + offset
		if a < 0 {
			a = 0
		}
		tryAdd(uint64(a), uint64(a)+rangeLen-1)
	}

	// In-gap queries: pick a random gap, place query inside.
	target := nNear + nGap
	if len(gaps) > 0 {
		for i := 0; i < nGap*2 && len(queries) < target; i++ {
			g := gaps[rng.Intn(len(gaps))]
			gapLen := g.hi - g.lo + 1
			if gapLen == 0 {
				continue
			}
			a := g.lo + uint64(rng.Int63n(int64(gapLen)))
			b := a + rangeLen - 1
			if b > g.hi {
				b = g.hi
			}
			if b >= a {
				queries = append(queries, [2]uint64{a, b}) // guaranteed empty (inside gap)
			}
		}
	}

	// Uniform queries: random across span.
	target = count
	for i := 0; i < nUnif*2 && len(queries) < target; i++ {
		a := minK + uint64(rng.Int63n(int64(span)))
		tryAdd(a, a+rangeLen-1)
	}

	return queries
}

func TestTradeoff_SOSD_Books(t *testing.T) {
	const queryCount = 1 << 18
	path := sosdPath("books_200M_uint32")

	for _, n := range []int{1 << 16, 1 << 18, 1 << 20, 1 << 24} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			keys, err := loadSOSDUint32(path, n)
			if err != nil {
				t.Skipf("SOSD data not available: %v (run bench/sosd_data/download.sh)", err)
			}
			t.Logf("loaded %d keys from books_200M, range [%d, %d]", len(keys), keys[0], keys[len(keys)-1])

			runTradeoffBench(t, benchConfig{
				distName: "sosd_books",
				n:        n,
				keys:     keys,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
				},
			})
		})
	}
}

func TestTradeoff_SOSD_Facebook(t *testing.T) {
	const queryCount = 1 << 20
	path := sosdPath("fb_200M_uint64")

	for _, n := range []int{1 << 16, 1 << 18, 1 << 20, 1 << 24} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			keys, err := loadSOSDUint64(path, n)
			if err != nil {
				t.Skipf("SOSD data not available: %v (run bench/sosd_data/download.sh)", err)
			}
			t.Logf("loaded %d keys from fb_200M, range [%d, %d]", len(keys), keys[0], keys[len(keys)-1])

			runTradeoffBench(t, benchConfig{
				distName: "sosd_fb",
				n:        n,
				keys:     keys,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
				},
			})
		})
	}
}

func TestTradeoff_SOSD_Wiki(t *testing.T) {
	const queryCount = 1 << 18
	path := sosdPath("wiki_ts_200M_uint64")

	for _, n := range []int{1 << 16, 1 << 18, 1 << 20, 1 << 24} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			keys, err := loadSOSDUint64(path, n)
			if err != nil {
				t.Skipf("SOSD data not available: %v (run bench/sosd_data/download.sh)", err)
			}
			t.Logf("loaded %d keys from wiki_ts, range [%d, %d]", len(keys), keys[0], keys[len(keys)-1])

			runTradeoffBench(t, benchConfig{
				distName: "sosd_wiki",
				n:        n,
				keys:     keys,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
				},
			})
		})
	}
}

func TestTradeoff_SOSD_OSM(t *testing.T) {
	const queryCount = 1 << 18
	path := sosdPath("osm_cellids_800M_uint64")

	for _, n := range []int{1 << 16, 1 << 18, 1 << 20, 1 << 24} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			keys, err := loadSOSDUint64(path, n)
			if err != nil {
				t.Skipf("SOSD data not available: %v (run bench/sosd_data/download.sh)", err)
			}
			t.Logf("loaded %d keys from osm_cellids, range [%d, %d]", len(keys), keys[0], keys[len(keys)-1])

			runTradeoffBench(t, benchConfig{
				distName: "sosd_osm",
				n:        n,
				keys:     keys,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
				},
			})
		})
	}
}
