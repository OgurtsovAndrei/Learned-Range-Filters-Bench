package bench_test

import (
	"Thesis/emptiness/approx/hybrid/are_dbscan"
	"Thesis/emptiness/approx/hybrid/are_greedy"
	"fmt"
	"sort"
	"testing"
)

func TestVerifyClusterCounts(t *testing.T) {
	// Generate some clustered data.
	// 1M keys total. 10 clusters of 10k keys each, very dense.
	// Remaining 900k keys are uniform.
	n := 1000000
	keys := make([]uint64, 0, n)

	// 10 clusters
	for c := 0; c < 10; c++ {
		center := uint64(c+1) * (1 << 60)
		for i := 0; i < 10000; i++ {
			keys = append(keys, center+uint64(i)*10) // spread = 100k
		}
	}

	// 900k uniform keys
	for i := 0; i < 900000; i++ {
		keys = append(keys, uint64(i)*(1<<40)+123456789)
	}

	// Sort and dedupe
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	j := 0
	for i := 1; i < len(keys); i++ {
		if keys[i] != keys[j] {
			j++
			keys[j] = keys[i]
		}
	}
	keys = keys[:j+1]

	K := uint32(24)

	// Create some queries: 1000 empty queries INSIDE the first cluster.
	center0 := uint64(1) << 60
	queries := [][2]uint64{}
	for i := 0; i < 1000; i++ {
		q := center0 + uint64(i)*10 + 5
		queries = append(queries, [2]uint64{q, q})
	}

	t.Run("Scan-ARE", func(t *testing.T) {
		f, err := are_dbscan.NewHybridScanARE(keys, 64, are_dbscan.Config{K: K})
		if err != nil {
			t.Fatal(err)
		}
		nc, nf, _ := f.Stats()

		// Use reflection or just trust my stats if I add them.
		// Since I can't change the code, I'll just check the first cluster if possible.
		// Wait, I can't access clusters field.

		hits := 0
		for _, q := range queries {
			if !f.IsEmpty(q[0], q[1]) {
				hits++
			}
		}
		fpr := float64(hits) / float64(len(queries))
		fmt.Printf("Scan-ARE: clusters=%d, fallback_keys=%d, FPR=%.4f\n", nc, nf, fpr)
	})

	t.Run("Greedy-ARE", func(t *testing.T) {
		f, err := are_greedy.NewGreedyScanARE(keys, 64, are_greedy.Config{K: K})
		if err != nil {
			t.Fatal(err)
		}
		nc, nf, _ := f.Stats()
		hits := 0
		for _, q := range queries {
			if !f.IsEmpty(q[0], q[1]) {
				hits++
			}
		}
		fpr := float64(hits) / float64(len(queries))
		fmt.Printf("Greedy-ARE: clusters=%d, fallback_keys=%d, FPR=%.4f\n", nc, nf, fpr)
	})
}
