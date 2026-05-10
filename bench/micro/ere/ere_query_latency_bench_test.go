package ere_test

import (
	"Thesis/emptiness/exact/ere"
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"testing"
)

// ereQueryLatencyFixture pre-builds the ERE structure and pre-generates query
// endpoints so the benchmark harness can re-run the timed loop at different
// b.N without re-loading data or rebuilding the ERE.
type ereQueryLatencyFixture struct {
	actualN int
	ere     *ere.ExactRangeEmptiness
	queryA  []uint64
	queryB  []uint64
}

func BenchmarkEREQueryLatency_Distributions(b *testing.B) {
	type dataset struct {
		name string
		load func(n int) ([]uint64, error)
	}

	datasets := []dataset{
		{"uniform", func(n int) ([]uint64, error) {
			rng := rand.New(rand.NewSource(42))
			keys := make([]uint64, 0, n)
			seen := make(map[uint64]struct{}, n)
			for len(keys) < n {
				v := rng.Uint64()
				if _, ok := seen[v]; !ok {
					seen[v] = struct{}{}
					keys = append(keys, v)
				}
			}
			sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
			return keys, nil
		}},
		{"clustered", func(n int) ([]uint64, error) {
			// Lightweight clustered generator: 8 clusters, keys sampled as
			// (center + small_offset) without a dedup map. Collisions are
			// resolved by rejecting duplicates post-sort (deterministic, fast).
			return generateLightClustered(n, 8, 42)
		}},
		{"sosd_fb", func(n int) ([]uint64, error) {
			return bucketLoadSOSD64(bucketSOSDPath("fb_200M_uint64"), n)
		}},
		{"sosd_wiki", func(n int) ([]uint64, error) {
			return bucketLoadSOSD64(bucketSOSDPath("wiki_ts_200M_uint64"), n)
		}},
		{"sosd_osm", func(n int) ([]uint64, error) {
			return bucketLoadSOSD64(bucketSOSDPath("osm_cellids_800M_uint64"), n)
		}},
		{"sosd_books", func(n int) ([]uint64, error) {
			return bucketLoadSOSD32(bucketSOSDPath("books_200M_uint32"), n)
		}},
	}

	ns := []int{1 << 20, 1 << 22, 1 << 24, 1 << 26}

	for _, ds := range datasets {
		for _, n := range ns {
			dsName := ds.name
			reqN := n
			loader := ds.load
			name := fmt.Sprintf("dist=%s/n=%d", dsName, reqN)

			// Prepare fixture outside b.Run so the load/build happens exactly once
			// per (dist, n). Failures are recorded in fixtureErr and surfaced via b.Skip.
			var fx *ereQueryLatencyFixture
			var fixtureErr error

			fx, fixtureErr = buildEREQueryLatencyFixture(loader, reqN)
			if fixtureErr != nil {
				fmt.Printf("[setup] %s: skipped: %v\n", name, fixtureErr)
			} else {
				fmt.Printf("[setup] %s: actualN=%d ready\n", name, fx.actualN)
			}

			b.Run(name, func(b *testing.B) {
				if fixtureErr != nil {
					b.Logf("skip %s: %v", name, fixtureErr)
					b.Skip("fixture build failed")
					return
				}
				b.Logf("dist=%s requested_n=%d actualN=%d", dsName, reqN, fx.actualN)

				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					idx := i % len(fx.queryA)
					_ = fx.ere.IsEmpty(fx.queryA[idx], fx.queryB[idx])
				}
				b.StopTimer()
			})

			// Release fixture after benchmark completes.
			if fx != nil {
				fx.ere = nil
				fx.queryA = nil
				fx.queryB = nil
			}
			fx = nil
			runtime.GC()
		}
	}
}

// generateLightClustered generates approximately n 64-bit keys distributed
// across numClusters Gaussian clusters. Keys are sorted and deduplicated via
// a linear pass over the sorted array (O(n log n) sort, no hash map), so it
// scales to n >= 64M without the memory/time overhead of GenerateClusterDistribution.
func generateLightClustered(n int, numClusters int, seed int64) ([]uint64, error) {
	rng := rand.New(rand.NewSource(seed))
	keys := make([]uint64, 0, n)

	// Cluster weights: exponential (Dirichlet-like).
	weights := make([]float64, numClusters)
	var wSum float64
	for i := range weights {
		weights[i] = rng.ExpFloat64()
		wSum += weights[i]
	}

	sizes := make([]int, numClusters)
	assigned := 0
	for i := range sizes {
		sizes[i] = int(weights[i] / wSum * float64(n))
		assigned += sizes[i]
	}
	sizes[numClusters-1] += n - assigned

	for c := 0; c < numClusters; c++ {
		center := rng.Uint64()
		stddev := float64(uint64(1) << (20 + rng.Intn(10)))
		for i := 0; i < sizes[c]; i++ {
			offset := int64(rng.NormFloat64() * stddev)
			var key uint64
			if offset >= 0 {
				key = center + uint64(offset)
			} else {
				key = center - uint64(-offset)
			}
			keys = append(keys, key)
		}
	}

	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })

	// Deduplicate in-place.
	j := 0
	for i := 1; i < len(keys); i++ {
		if keys[i] != keys[j] {
			j++
			keys[j] = keys[i]
		}
	}
	if len(keys) > 0 {
		keys = keys[:j+1]
	}
	return keys, nil
}

func buildEREQueryLatencyFixture(load func(n int) ([]uint64, error), n int) (*ereQueryLatencyFixture, error) {
	keys, err := load(n)
	if err != nil {
		return nil, fmt.Errorf("load: %w", err)
	}
	actualN := len(keys)
	if actualN == 0 {
		return nil, fmt.Errorf("zero keys after load")
	}

	ereStruct, err := ere.NewExactRangeEmptiness(keys, 64)
	if err != nil {
		return nil, fmt.Errorf("ere build: %w", err)
	}

	const numQueries = 100
	queryA := make([]uint64, numQueries)
	queryB := make([]uint64, numQueries)
	for i := 0; i < numQueries; i++ {
		idxA := i * actualN / numQueries
		idxB := (i + 1) * actualN / numQueries
		if idxA >= actualN {
			idxA = actualN - 1
		}
		if idxB >= actualN {
			idxB = actualN - 1
		}
		queryA[i] = keys[idxA]
		queryB[i] = keys[idxB]
	}

	keys = nil
	runtime.GC()

	return &ereQueryLatencyFixture{
		actualN: actualN,
		ere:     ereStruct,
		queryA:  queryA,
		queryB:  queryB,
	}, nil
}
