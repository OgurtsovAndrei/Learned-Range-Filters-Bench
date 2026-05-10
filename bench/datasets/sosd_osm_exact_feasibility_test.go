package datasets_test

import (
	"fmt"
	"math/bits"
	"os"
	"sort"
	"strconv"
	"testing"

	"Thesis-bench-industry/bench/datasets"
)

// TestSOSD_OSM_ExactModeFeasibility answers: are there m-key consecutive
// windows in sosd_osm dense enough that AdaptiveARE would pick exact mode
// (M = bits.Len64(localSpan) <= K) inside such a window?
//
// Without this, density-based clustering buys nothing on osm: even when a
// cluster is detected, AdaptiveARE flips to SODA-mode and produces no
// segmentation gain over plain SODA fallback.
//
// Set OSM_FEAS_N=2^28 (or any other power of 2) to test the user's
// hypothesis that sampling more keys reduces local span enough to flip
// more windows into exact mode.
func TestSOSD_OSM_ExactModeFeasibility(t *testing.T) {
	maxKeys := 1 << 24
	if v := os.Getenv("OSM_FEAS_N"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n <= 0 {
			t.Fatalf("bad OSM_FEAS_N=%q", v)
		}
		maxKeys = n
	}

	r := &datasets.SOSDReader{
		Path:    sosdPath("osm_cellids_800M_uint64"),
		Label:   "sosd_osm",
		KeyType: datasets.SOSDUint64,
		MaxKeys: maxKeys,
	}
	keys, err := r.Keys()
	if err != nil {
		t.Skipf("osm_cellids_800M_uint64 not available: %v", err)
	}
	n := len(keys)
	if n < 2 {
		t.Fatalf("not enough keys: %d", n)
	}

	span := keys[n-1] - keys[0]
	meanGap := float64(span) / float64(n-1)

	gaps := make([]uint64, n-1)
	for i := 1; i < n; i++ {
		gaps[i-1] = keys[i] - keys[i-1]
	}
	sortedGaps := append([]uint64(nil), gaps...)
	sort.Slice(sortedGaps, func(i, j int) bool { return sortedGaps[i] < sortedGaps[j] })
	pct := func(p float64) uint64 {
		idx := int(p * float64(len(sortedGaps)-1))
		return sortedGaps[idx]
	}

	fmt.Printf("\n=== sosd_osm exact-mode feasibility (n=%d) ===\n", n)
	fmt.Printf("global span=%d (~2^%.1f) → M_global=%d\n",
		span, log2u64(span), bits.Len64(span))
	fmt.Printf("mean gap=%.0f (~2^%.1f), p50=%d, p90=%d, p99=%d, p99.9=%d, max=%d\n",
		meanGap, log2u64(uint64(meanGap)),
		pct(0.5), pct(0.9), pct(0.99), pct(0.999), sortedGaps[len(sortedGaps)-1])

	// Sliding window: smallest local span over m consecutive keys.
	winSizes := []int{256, 1024, 4096, 16384, 65536, 262144}
	type winStat struct {
		m         int
		minSpan   uint64
		minIdx    int
		minMbits  int
	}
	stats := make([]winStat, 0, len(winSizes))
	for _, m := range winSizes {
		if m >= n {
			continue
		}
		var minSpan uint64 = ^uint64(0)
		var minIdx int
		for i := 0; i+m < n; i++ {
			ls := keys[i+m] - keys[i]
			if ls < minSpan {
				minSpan = ls
				minIdx = i
			}
		}
		stats = append(stats, winStat{m, minSpan, minIdx, bits.Len64(minSpan)})
	}

	fmt.Printf("\nDensest m-window in osm:\n")
	fmt.Printf("%-8s %-14s %-10s %-10s %-10s %-12s\n",
		"m", "minSpan", "M", "log2", "atIdx", "frac of n")
	for _, s := range stats {
		fmt.Printf("%-8d %-14d %-10d %-10.2f %-10d %-12.4f%%\n",
			s.m, s.minSpan, s.minMbits, log2u64(s.minSpan),
			s.minIdx, 100*float64(s.minIdx)/float64(n))
	}

	// For each candidate K, count how many m=256-windows have M <= K
	// (exact-mode-eligible at that K).
	fmt.Printf("\nFraction of non-overlapping m=256 windows with M ≤ K:\n")
	const m = 256
	if m < n {
		spansM := make([]uint64, 0, n/m)
		for i := 0; i+m < n; i += m {
			spansM = append(spansM, keys[i+m]-keys[i])
		}
		sort.Slice(spansM, func(i, j int) bool { return spansM[i] < spansM[j] })
		fmt.Printf("(total non-overlapping windows: %d)\n", len(spansM))
		for _, K := range []uint32{20, 24, 28, 32, 36, 40, 44, 48} {
			thr := uint64(1) << K
			cnt := 0
			for _, s := range spansM {
				if s <= thr {
					cnt++
				}
			}
			fmt.Printf("  K=%-3d (2^K=%-20d): %-7d / %-7d eligible (%.4f%%)\n",
				K, thr, cnt, len(spansM),
				100*float64(cnt)/float64(len(spansM)))
		}
		fmt.Printf("\nm=256 window-span percentiles (smallest = densest):\n")
		for _, p := range []float64{0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999} {
			idx := int(p * float64(len(spansM)-1))
			s := spansM[idx]
			fmt.Printf("  p%-7.4f span = %-14d  (M=%d, ~2^%.1f)\n",
				p, s, bits.Len64(s), log2u64(s))
		}
	}

	// Same for larger m to see how cluster size affects feasibility.
	fmt.Printf("\nFraction of non-overlapping windows with M ≤ K, by m:\n")
	fmt.Printf("%-7s", "K\\m")
	for _, mm := range []int{256, 1024, 4096, 16384} {
		fmt.Printf(" %-9d", mm)
	}
	fmt.Println()
	for _, K := range []uint32{32, 36, 40, 44, 48} {
		fmt.Printf("K=%-5d", K)
		thr := uint64(1) << K
		for _, mm := range []int{256, 1024, 4096, 16384} {
			if mm >= n {
				fmt.Printf(" %-9s", "n/a")
				continue
			}
			total := 0
			cnt := 0
			for i := 0; i+mm < n; i += mm {
				total++
				if keys[i+mm]-keys[i] <= thr {
					cnt++
				}
			}
			pctStr := "0.000%"
			if total > 0 {
				pctStr = fmt.Sprintf("%.3f%%", 100*float64(cnt)/float64(total))
			}
			fmt.Printf(" %-9s", pctStr)
		}
		fmt.Println()
	}
}
