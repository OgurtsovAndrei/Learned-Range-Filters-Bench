package ere_test

import (
	"Thesis/emptiness/approx/are_soda_hash"
	"Thesis/emptiness/exact"
	"Thesis/testutils"
	"fmt"
	"math/rand"
	"sort"
	"testing"
)

// TestEREBucketSecondMoment_SodaARE measures the time-weighted bucket size of
// the ERE backend inside SodaARE for several distributions and range lengths.
//
// For a partition of the universe into B blocks where the b-th block holds k_b
// keys, the standard "uniform" average bucket size is (Sum k_b) / M over the M
// non-empty blocks. Non-empty range queries are not uniform over blocks though:
// if every stored key is equally likely to be a query target, then the
// probability that a query lands in block b is k_b / n. The expected per-query
// bucket-search cost (in keys touched, ignoring constants) is therefore
//
//	E[bucket size touched by query] = Sum_b (k_b / n) * k_b = (1 / n) * Sum_b k_b^2
//
// the second moment of bucket occupancy normalised by n. We also report
// key-weighted percentiles X_p of the non-empty bucket-size distribution: the
// smallest k such that buckets of size <= k collectively hold at least p * n
// keys. This answers "what fraction of *keys* (and therefore queries that hit
// data) live in buckets of size <= k", which is the relevant quantity for
// query-time tail analysis.
func TestEREBucketSecondMoment_SodaARE(t *testing.T) {
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

	fmt.Printf("%-12s %-6s %-10s %-10s %-8s %-8s %-8s %-8s %-8s %-10s %-14s %-8s\n",
		"distribution", "L", "B", "M",
		"X_50", "X_90", "X_95", "X_99",
		"max", "avg_unif", "second_moment", "ratio")
	fmt.Println("--------------------------------------------------------------------------------------------------------------------")

	for _, ds := range datasets {
		keys, err := ds.load()
		if err != nil {
			t.Logf("skip %s: %v", ds.name, err)
			continue
		}

		for _, L := range rangeLens {
			name := fmt.Sprintf("%s/L=%d", ds.name, L)
			t.Run(name, func(t *testing.T) {
				are, err := are_soda_hash.NewSodaAREWithBackend(keys, L, epsilon, exact.VariantOneD)
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
						// smallest cum >= p * totalKeys
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

				fmt.Printf("%-12s %-6d %-10d %-10d %-8d %-8d %-8d %-8d %-8d %-10.4f %-14.4f %-8.4f\n",
					ds.name, L, stats.NumBlocks, stats.NonEmptyBlocks,
					x50, x90, x95, x99,
					stats.MaxKeysInBlock, avgUniform, secondMoment, ratio)
			})
		}
	}
}
