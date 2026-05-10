package ere_test

import (
	"Thesis/emptiness/approx/are_soda_hash"
	"Thesis/testutils"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"testing"
	"time"
)

// TestEREBucketSecondMoment_SodaARE_Large is the large-scale companion to
// TestEREBucketSecondMoment_SodaARE. It runs the same bucket second-moment
// metric extraction at the maximum n that fits within ~50 GB RSS on a 64 GB
// machine, using the destructive uint64 build path NewSodaAREUint64InPlace.
//
// Per-distribution n:
//   - uniform, clustered : 2^30 (~1.07G keys; only L=16 for uniform)
//   - sosd_fb, sosd_wiki, sosd_books : 2^27 (~134M keys)
//   - sosd_osm : 2^29 (~537M keys)
//
// Each combo is run sequentially; the keys slice and the filter are released
// between combos to allow the GC to reclaim memory before the next build.
func TestEREBucketSecondMoment_SodaARE_Large(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large second-moment test in -short mode")
	}

	const epsilon = 0.01
	const memoryBudgetBytes uint64 = 50 * 1024 * 1024 * 1024 // 50 GB

	type combo struct {
		dist string
		n    int
		L    uint64
		load func(n int) ([]uint64, error)
	}

	uniformLoader := func(n int) ([]uint64, error) {
		rng := rand.New(rand.NewSource(42))
		keys := make([]uint64, n)
		for i := 0; i < n; i++ {
			keys[i] = rng.Uint64()
		}
		sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
		w := 1
		for i := 1; i < len(keys); i++ {
			if keys[i] != keys[i-1] {
				keys[w] = keys[i]
				w++
			}
		}
		return keys[:w], nil
	}

	clusteredLoader := func(n int) ([]uint64, error) {
		// Memory-efficient clustered generator: 10% uniform + 90% Gaussian
		// clusters, written directly into a single slice (no seen-map). Final
		// sort+dedupe drops any collisions; for n=2^30 with 8 clusters of
		// stddev ~ 2^29 the duplicate rate is tiny.
		const numClusters = 8
		const unifFrac = 0.1
		rng := rand.New(rand.NewSource(42))

		nUnif := int(float64(n) * unifFrac)
		nClust := n - nUnif

		weights := make([]float64, numClusters)
		var wSum float64
		for i := range weights {
			weights[i] = rng.ExpFloat64()
			wSum += weights[i]
		}
		clusterSizes := make([]int, numClusters)
		assigned := 0
		for i := range clusterSizes {
			clusterSizes[i] = int(weights[i] / wSum * float64(nClust))
			assigned += clusterSizes[i]
		}
		clusterSizes[numClusters-1] += nClust - assigned

		keys := make([]uint64, 0, n)
		for i := 0; i < nUnif; i++ {
			keys = append(keys, rng.Uint64())
		}
		for c := 0; c < numClusters; c++ {
			center := rng.Uint64()
			stddev := float64(uint64(1) << (20 + rng.Intn(10)))
			generated := 0
			for generated < clusterSizes[c] {
				v := testutils.SampleGaussian(center, stddev, rng)
				if v == 0 && center != 0 {
					continue
				}
				keys = append(keys, v)
				generated++
			}
		}
		sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
		w := 1
		for i := 1; i < len(keys); i++ {
			if keys[i] != keys[i-1] {
				keys[w] = keys[i]
				w++
			}
		}
		return keys[:w], nil
	}

	combos := []combo{
		{"uniform", 1 << 30, 16, uniformLoader},

		{"clustered", 1 << 30, 16, clusteredLoader},
		{"clustered", 1 << 30, 256, clusteredLoader},
		{"clustered", 1 << 30, 4096, clusteredLoader},

		{"sosd_fb", 1 << 27, 16, func(n int) ([]uint64, error) {
			return bucketLoadSOSD64(bucketSOSDPath("fb_200M_uint64"), n)
		}},
		{"sosd_fb", 1 << 27, 256, func(n int) ([]uint64, error) {
			return bucketLoadSOSD64(bucketSOSDPath("fb_200M_uint64"), n)
		}},
		{"sosd_fb", 1 << 27, 4096, func(n int) ([]uint64, error) {
			return bucketLoadSOSD64(bucketSOSDPath("fb_200M_uint64"), n)
		}},

		{"sosd_wiki", 1 << 27, 16, func(n int) ([]uint64, error) {
			return bucketLoadSOSD64(bucketSOSDPath("wiki_ts_200M_uint64"), n)
		}},
		{"sosd_wiki", 1 << 27, 256, func(n int) ([]uint64, error) {
			return bucketLoadSOSD64(bucketSOSDPath("wiki_ts_200M_uint64"), n)
		}},
		{"sosd_wiki", 1 << 27, 4096, func(n int) ([]uint64, error) {
			return bucketLoadSOSD64(bucketSOSDPath("wiki_ts_200M_uint64"), n)
		}},

		{"sosd_books", 1 << 27, 16, func(n int) ([]uint64, error) {
			return bucketLoadSOSD32(bucketSOSDPath("books_200M_uint32"), n)
		}},
		{"sosd_books", 1 << 27, 256, func(n int) ([]uint64, error) {
			return bucketLoadSOSD32(bucketSOSDPath("books_200M_uint32"), n)
		}},
		{"sosd_books", 1 << 27, 4096, func(n int) ([]uint64, error) {
			return bucketLoadSOSD32(bucketSOSDPath("books_200M_uint32"), n)
		}},

		{"sosd_osm", 1 << 29, 16, func(n int) ([]uint64, error) {
			return bucketLoadSOSD64(bucketSOSDPath("osm_cellids_800M_uint64"), n)
		}},
		{"sosd_osm", 1 << 29, 256, func(n int) ([]uint64, error) {
			return bucketLoadSOSD64(bucketSOSDPath("osm_cellids_800M_uint64"), n)
		}},
		{"sosd_osm", 1 << 29, 4096, func(n int) ([]uint64, error) {
			return bucketLoadSOSD64(bucketSOSDPath("osm_cellids_800M_uint64"), n)
		}},
	}

	fmt.Printf("%-12s %-12s %-6s %-12s %-12s %-10s %-10s %-10s %-10s %-10s %-12s %-16s %-10s %-10s\n",
		"distribution", "n", "L", "B", "M",
		"X_50", "X_90", "X_95", "X_99",
		"max", "avg_unif", "second_moment", "ratio", "wall_s")
	fmt.Println("------------------------------------------------------------------------------------------------------------------------------------------------------------")

	for _, cb := range combos {
		name := fmt.Sprintf("%s/n=%d/L=%d", cb.dist, cb.n, cb.L)
		t.Run(name, func(t *testing.T) {
			start := time.Now()

			// Memory pre-flight: refuse if we already exceed budget.
			var ms runtime.MemStats
			runtime.ReadMemStats(&ms)
			if ms.HeapAlloc > memoryBudgetBytes {
				t.Skipf("heap already at %.1f GB before combo (budget %.1f GB)",
					float64(ms.HeapAlloc)/1e9, float64(memoryBudgetBytes)/1e9)
			}

			keys, err := cb.load(cb.n)
			if err != nil {
				t.Skipf("dataset load failed: %v", err)
			}
			actualN := len(keys)

			// K = ceil(log2(n * L / epsilon)), matching NewSodaARE.
			rTarget := float64(actualN) * float64(cb.L) / epsilon
			K := uint32(math.Ceil(math.Log2(rTarget)))
			if K > 64 {
				keys = nil
				runtime.GC()
				t.Skipf("K=%d exceeds 64 bits", K)
			}

			are, err := are_soda_hash.NewSodaAREUint64InPlace(keys, K, int64(cb.L))
			// keys is now consumed; release the reference so the slice can be GC'd
			// once SortAndDedupUint64's slice (a sub-slice of keys) is also dropped
			// via `are`.
			keys = nil
			if err != nil {
				t.Fatalf("build failed: %v", err)
			}

			stats := are.EREStats()
			sizes := are.ERENonEmptyBlockSizes()
			if len(sizes) == 0 {
				t.Fatalf("no non-empty buckets reported")
			}

			sort.Ints(sizes)
			totalKeys := uint64(0)
			for _, s := range sizes {
				totalKeys += uint64(s)
			}

			keyWeightedPct := func(p float64) int {
				threshold := uint64(0)
				if totalKeys > 0 {
					thresholdF := p * float64(totalKeys)
					threshold = uint64(thresholdF)
					if float64(threshold) < thresholdF {
						threshold++
					}
				}
				var cum uint64
				for _, s := range sizes {
					cum += uint64(s)
					if cum >= threshold {
						return s
					}
				}
				return sizes[len(sizes)-1]
			}

			x50 := keyWeightedPct(0.50)
			x90 := keyWeightedPct(0.90)
			x95 := keyWeightedPct(0.95)
			x99 := keyWeightedPct(0.99)

			avgUniform := stats.AvgKeysPerBlock
			secondMoment := float64(stats.SumSquaredKeys) / float64(totalKeys)
			ratio := 0.0
			if avgUniform > 0 {
				ratio = secondMoment / avgUniform
			}

			elapsed := time.Since(start)
			fmt.Printf("%-12s %-12d %-6d %-12d %-12d %-10d %-10d %-10d %-10d %-10d %-12.4f %-16.4f %-10.4f %-10.2f\n",
				cb.dist, actualN, cb.L, stats.NumBlocks, stats.NonEmptyBlocks,
				x50, x90, x95, x99,
				stats.MaxKeysInBlock, avgUniform, secondMoment, ratio,
				elapsed.Seconds())

			if elapsed > 30*time.Minute {
				t.Logf("WARNING: combo %s took %s (>30m budget)", name, elapsed)
			}

			// Free everything before next combo.
			sizes = nil
			are = nil
			runtime.GC()
		})
	}
}
