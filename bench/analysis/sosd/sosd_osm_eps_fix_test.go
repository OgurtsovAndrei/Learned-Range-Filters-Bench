//go:build heavy

package sosd_test

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"testing"
	"time"

	"Thesis-bench-industry/bench/datasets"
	are_hybrid_scan "Thesis/emptiness/approx/hybrid/are_dbscan"
	are_greedy_scan "Thesis/emptiness/approx/hybrid/are_greedy"
	"Thesis/emptiness/approx/hybrid/hybridutil"
	"Thesis/emptiness/approx/are_soda_hash"
	"Thesis/testutils"
	"Thesis-bench-industry/thirdparty/grafite"
)

func epsFixDataPath(name string) string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), "sosd_data", name)
}

// epsFixRandUint64Below returns a random uint64 in [0, n). Tolerates
// n > 2^63 (where Int63n would panic).
func epsFixRandUint64Below(rng *rand.Rand, n uint64) uint64 {
	if n == 0 {
		return 0
	}
	if n <= 1<<63 {
		return uint64(rng.Int63n(int64(n)))
	}
	return rng.Uint64() % n
}

// epsFixSmartQueries returns "smart" empty range queries: 50% near-key,
// 30% in-gap, 20% uniform across the span. All guaranteed empty.
func epsFixSmartQueries(keys []uint64, count int, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	n := len(keys)
	minK, maxK := keys[0], keys[n-1]
	span := maxK - minK
	if span == 0 {
		return nil
	}
	nNear := count * 50 / 100
	nGap := count * 30 / 100

	type gap struct{ lo, hi uint64 }
	gaps := make([]gap, 0, 1<<16)
	for i := 0; i < n-1; i++ {
		if keys[i+1]-keys[i] > rangeLen {
			gaps = append(gaps, gap{keys[i] + 1, keys[i+1] - rangeLen})
		}
		if len(gaps) >= 1<<20 {
			break
		}
	}

	queries := make([][2]uint64, 0, count)

	// Near-key: pick a random key, offset by [-5L, +5L].
	for tries := 0; tries < nNear*3 && len(queries) < nNear; tries++ {
		k := keys[rng.Intn(n)]
		off := rng.Int63n(int64(rangeLen)*10) - int64(rangeLen)*5
		var a uint64
		if off < 0 {
			d := uint64(-off)
			if d > k {
				continue
			}
			a = k - d
		} else {
			a = k + uint64(off)
		}
		b := a + rangeLen - 1
		idx := sort.Search(n, func(i int) bool { return keys[i] >= a })
		if idx < n && keys[idx] <= b {
			if keys[idx] == 0 || keys[idx]-1 < a {
				continue
			}
			b = keys[idx] - 1
		}
		queries = append(queries, [2]uint64{a, b})
	}

	// In-gap.
	target := nNear + nGap
	if len(gaps) > 0 {
		for tries := 0; tries < nGap*3 && len(queries) < target; tries++ {
			g := gaps[rng.Intn(len(gaps))]
			gapLen := g.hi - g.lo + 1
			if gapLen == 0 {
				continue
			}
			a := g.lo + epsFixRandUint64Below(rng, gapLen)
			b := a + rangeLen - 1
			if b > g.hi {
				b = g.hi
			}
			if b >= a {
				queries = append(queries, [2]uint64{a, b})
			}
		}
	}

	// Uniform.
	for tries := 0; tries < count*3 && len(queries) < count; tries++ {
		a := minK + epsFixRandUint64Below(rng, span)
		b := a + rangeLen - 1
		idx := sort.Search(n, func(i int) bool { return keys[i] >= a })
		if idx < n && keys[idx] <= b {
			if keys[idx] == 0 || keys[idx]-1 < a {
				continue
			}
			b = keys[idx] - 1
		}
		queries = append(queries, [2]uint64{a, b})
	}
	return queries
}

// TestEpsFix_HybridScan_OSM_Full benchmarks Hybrid-Scan-ARE on the entire
// OSM dataset (800M keys, span=2^59.7) with smart "near-key + in-gap"
// queries. The same test source runs against either eps formula via the
// Thesis submodule's checked-out commit; flipping the formula in
// hybrid_scan_are.go and rerunning produces the before/after pair.
//
// Set OSM_FULL_N=<int> to override key count (default: 800M = entire dataset).
// Set OSM_FULL_K=<comma-sep> to override K sweep (default: 32,36,40,44).
// Set OSM_FULL_QUERIES=<int> to override query count (default: 30000).
func TestEpsFix_HybridScan_OSM_Full(t *testing.T) {
	maxKeys := 0 // 0 → all 800M
	if v := os.Getenv("OSM_FULL_N"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n <= 0 {
			t.Fatalf("bad OSM_FULL_N=%q", v)
		}
		maxKeys = n
	}
	queryCount := 30_000
	if v := os.Getenv("OSM_FULL_QUERIES"); v != "" {
		q, err := strconv.Atoi(v)
		if err != nil || q <= 0 {
			t.Fatalf("bad OSM_FULL_QUERIES=%q", v)
		}
		queryCount = q
	}
	Ks := []uint32{40, 44, 48, 52}
	if v := os.Getenv("OSM_FULL_K"); v != "" {
		Ks = nil
		for _, s := range splitCSV(v) {
			k, err := strconv.Atoi(s)
			if err != nil || k <= 0 {
				t.Fatalf("bad OSM_FULL_K=%q", v)
			}
			Ks = append(Ks, uint32(k))
		}
	}

	t0 := time.Now()
	r := &datasets.SOSDReader{
		Path:    epsFixDataPath("osm_cellids_800M_uint64"),
		Label:   "sosd_osm",
		KeyType: datasets.SOSDUint64,
		MaxKeys: maxKeys,
	}
	keys, err := r.Keys()
	if err != nil {
		t.Skipf("osm data unavailable: %v", err)
	}
	n := len(keys)
	span := keys[n-1] - keys[0]
	t.Logf("OSM loaded: n=%d span=2^%.1f (took %s)",
		n, math.Log2(float64(span)), time.Since(t0).Round(time.Second))

	const rangeLen = uint64(128)
	tQ := time.Now()
	rng := rand.New(rand.NewSource(0xCAFE))
	queries := epsFixSmartQueries(keys, queryCount, rangeLen, rng)
	t.Logf("smart queries: %d (took %s)",
		len(queries), time.Since(tQ).Round(time.Second))

	// Use 64-bit keys directly — our structures (Hybrid-Scan, Greedy-Scan,
	// SODA, Grafite) all support 64-bit keys natively. SNARF (excluded here)
	// is the only filter that needs the 60-bit mask.
	keyBits := uint32(64)

	fmt.Printf("\n%-18s %-6s %-10s %-10s %-12s %-9s %-10s\n",
		"filter", "K", "epsTarget", "#clust/—", "FPR", "BPK", "build")
	for _, K := range Ks {
		// epsilon target derived from K so Grafite can be sized comparably:
		//   K = ceil(log2(n*L/eps))  ⇒  eps = n*L / 2^K
		twoK := math.Exp2(float64(K))
		epsTarget := float64(len(keys)) * float64(rangeLen) / twoK
		if epsTarget > 0.5 {
			epsTarget = 0.5
		}

		// Scan-ARE (DBSCAN) — Trunc fallback
		tB := time.Now()
		hsT, err := are_hybrid_scan.NewHybridScanAREWithPolicy(keys, keyBits,
			are_hybrid_scan.ConfigWithPolicy{K: K, RangeLen: rangeLen, Policy: hybridutil.FallbackAlwaysTrunc{}})
		if err != nil {
			t.Errorf("K=%d Scan-ARE-Trunc build: %v", K, err)
		} else {
			buildDur := time.Since(tB)
			nc, nfb, nt := hsT.Stats()
			covPct := 100 * float64(nt-nfb) / float64(nt)
			fpr := testutils.MeasureFPR(keys, queries, hsT.IsEmpty)
			bpk := float64(hsT.SizeInBits()) / float64(len(keys))
			fmt.Printf("%-18s %-6d %-10.2e %-10s %-12.5e %-9.3f %-10s\n",
				"Scan-ARE-Trunc", K, epsTarget,
				fmt.Sprintf("%d/%.1f%%", nc, covPct),
				fpr, bpk, buildDur.Round(time.Second))
		}

		// Scan-ARE (DBSCAN) — SODA fallback
		tB = time.Now()
		hsS, err := are_hybrid_scan.NewHybridScanAREWithPolicy(keys, keyBits,
			are_hybrid_scan.ConfigWithPolicy{K: K, RangeLen: rangeLen, Policy: hybridutil.FallbackAlwaysSODA{}})
		if err != nil {
			t.Errorf("K=%d Scan-ARE-SODA build: %v", K, err)
		} else {
			buildDur := time.Since(tB)
			nc, nfb, nt := hsS.Stats()
			covPct := 100 * float64(nt-nfb) / float64(nt)
			fpr := testutils.MeasureFPR(keys, queries, hsS.IsEmpty)
			bpk := float64(hsS.SizeInBits()) / float64(len(keys))
			fmt.Printf("%-18s %-6d %-10.2e %-10s %-12.5e %-9.3f %-10s\n",
				"Scan-ARE-SODA", K, epsTarget,
				fmt.Sprintf("%d/%.1f%%", nc, covPct),
				fpr, bpk, buildDur.Round(time.Second))
		}

		// Greedy+Merge — Trunc fallback
		tB = time.Now()
		gsT, err := are_greedy_scan.NewGreedyScanAREWithPolicy(keys, keyBits,
			are_greedy_scan.ConfigWithPolicy{K: K, Policy: hybridutil.FallbackAlwaysTrunc{}})
		if err != nil {
			t.Errorf("K=%d Greedy+Merge-Trunc build: %v", K, err)
		} else {
			buildDur := time.Since(tB)
			nc, _, _ := gsT.Stats()
			fpr := testutils.MeasureFPR(keys, queries, gsT.IsEmpty)
			bpk := float64(gsT.SizeInBits()) / float64(len(keys))
			fmt.Printf("%-18s %-6d %-10.2e %-10s %-12.5e %-9.3f %-10s\n",
				"Greedy+Merge-Trunc", K, epsTarget, fmt.Sprintf("%d", nc),
				fpr, bpk, buildDur.Round(time.Second))
		}

		// Greedy+Merge — SODA fallback
		tB = time.Now()
		gsS, err := are_greedy_scan.NewGreedyScanAREWithPolicy(keys, keyBits,
			are_greedy_scan.ConfigWithPolicy{K: K, Policy: hybridutil.FallbackAlwaysSODA{}})
		if err != nil {
			t.Errorf("K=%d Greedy+Merge-SODA build: %v", K, err)
		} else {
			buildDur := time.Since(tB)
			nc, _, _ := gsS.Stats()
			fpr := testutils.MeasureFPR(keys, queries, gsS.IsEmpty)
			bpk := float64(gsS.SizeInBits()) / float64(len(keys))
			fmt.Printf("%-18s %-6d %-10.2e %-10s %-12.5e %-9.3f %-10s\n",
				"Greedy+Merge-SODA", K, epsTarget, fmt.Sprintf("%d", nc),
				fpr, bpk, buildDur.Round(time.Second))
		}

		// SODA baseline
		tB = time.Now()
		sd, err := are_soda_hash.NewSodaAREFromK(keys, K, int64(K)*1000003+int64(len(keys)))
		if err != nil {
			t.Errorf("K=%d soda build: %v", K, err)
		} else {
			buildDur := time.Since(tB)
			fpr := testutils.MeasureFPR(keys, queries, sd.IsEmpty)
			bpk := float64(sd.SizeInBits()) / float64(len(keys))
			fmt.Printf("%-18s %-6d %-10.2e %-10s %-12.5e %-9.3f %-10s\n",
				"SODA", K, epsTarget, "—",
				fpr, bpk, buildDur.Round(time.Second))
		}

		// Grafite (paper-faithful eps,L constructor)
		tB = time.Now()
		gf := grafite.NewWithEpsL(keys, epsTarget, rangeLen)
		buildDur := time.Since(tB)
		fpr := testutils.MeasureFPR(keys, queries, gf.IsEmpty)
		bpk := float64(gf.SizeInBits()) / float64(len(keys))
		fmt.Printf("%-18s %-6d %-10.2e %-10s %-12.5e %-9.3f %-10s\n",
			"Grafite", K, epsTarget, "—",
			fpr, bpk, buildDur.Round(time.Second))

		fmt.Println()
	}
}

func splitCSV(s string) []string {
	var out []string
	cur := ""
	for _, r := range s {
		if r == ',' {
			if cur != "" {
				out = append(out, cur)
			}
			cur = ""
		} else {
			cur += string(r)
		}
	}
	if cur != "" {
		out = append(out, cur)
	}
	return out
}
