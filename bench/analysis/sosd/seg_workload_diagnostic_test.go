package sosd_test

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"

	"Thesis/emptiness/approx/hybrid/hybridutil"

	"Thesis-bench-industry/bench/internal/keygen"
	"Thesis-bench-industry/bench/internal/querygen"
)

// loadSOSDSorted loads `n` SOSD keys, sorts ascending, and dedupes.
func loadSOSDSorted(t *testing.T, name string, n int) []uint64 {
	t.Helper()
	keys, err := keygen.LoadSOSDUint64(keygen.SOSDPath(name), n)
	if err != nil {
		t.Skipf("SOSD %s not available: %v", name, err)
		return nil
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	j := 0
	for i := 1; i < len(keys); i++ {
		if keys[i] != keys[j] {
			j++
			keys[j] = keys[i]
		}
	}
	return keys[:j+1]
}

func runWorkloadMatrix(t *testing.T, label string, keys []uint64) {
	const L = uint64(128)
	const eps = 1e-3

	workloads := []struct {
		name    string
		weights querygen.SmartMixWeights
	}{
		{"pure_in_gap", querygen.SmartMixWeights{NearKey: 0.0, InGap: 1.0, Uniform: 0.0}},
		{"smart_mix", querygen.SmartMixWeights{NearKey: 0.5, InGap: 0.3, Uniform: 0.2}},
		{"pure_near_key", querygen.SmartMixWeights{NearKey: 1.0, InGap: 0.0, Uniform: 0.0}},
		{"pure_uniform", querygen.SmartMixWeights{NearKey: 0.0, InGap: 0.0, Uniform: 1.0}},
	}
	policies := []struct {
		name string
		p    hybridutil.FallbackPolicy
	}{
		{"AlwaysSODA", hybridutil.FallbackAlwaysSODA{}},
		{"AlwaysTrunc", hybridutil.FallbackAlwaysTrunc{}},
		{"InGapFPR", hybridutil.FallbackInGapFPR{Epsilon: eps}},
	}

	fmt.Printf("\n=== %s (n=%d, L=%d, ε=%g) ===\n", label, len(keys), L, eps)
	fmt.Printf("%-16s | %-12s | %10s | %12s | %8s | %10s\n", "workload", "policy", "BPK", "FPR", "nC", "nF")
	fmt.Println("--------------------------------------------------------------------------------")
	for _, w := range workloads {
		rng := rand.New(rand.NewSource(42))
		queries := querygen.GenerateSmartQueriesWeighted(keys, 100_000, L, w.weights, rng)
		if len(queries) == 0 {
			fmt.Printf("%-16s | (no queries generated)\n", w.name)
			continue
		}
		for _, pol := range policies {
			r := runSegOnce(t, keys, queries, L, eps, pol.p, pol.name)
			fmt.Printf("%-16s | %-12s | %10.2f | %12.3e | %8d | %10d\n",
				w.name, pol.name, r.bpk, r.fpr, r.nClusters, r.nFallback)
		}
		fmt.Println()
	}
}

// TestSegFallbackPolicy_SOSD_Diagnostic runs the same matrix on all 4 SOSD
// distributions (FB, Wiki, OSM, Books) at n=2^20.
func TestSegFallbackPolicy_SOSD_Diagnostic(t *testing.T) {
	const n = 1 << 20
	datasets := []struct {
		label string
		name  string
	}{
		{"SOSD/fb", "fb_200M_uint64"},
		{"SOSD/wiki", "wiki_ts_200M_uint64"},
		{"SOSD/osm", "osm_cellids_800M_uint64"},
		{"SOSD/books", "books_800M_uint64"},
	}
	for _, d := range datasets {
		keys := loadSOSDSorted(t, d.name, n)
		if keys == nil {
			continue
		}
		runWorkloadMatrix(t, d.label, keys)
	}
}

// TestSegFallbackPolicy_WorkloadDiagnostic runs the same three policies on
// three different query mixes on uniform synthetic data:
//   - pure_in_gap      (InGap=1.0)   ← our formula targets this exactly
//   - smart_mix        (50/30/20)    ← what the main bench uses
//   - pure_near_key    (NearKey=1.0) ← stress test, formula does NOT cover it
//
// Goal: show that on pure_in_gap Trunc stays under ε; the FPR explosion on
// smart_mix comes from the near-key bucket, not the formula.
func TestSegFallbackPolicy_WorkloadDiagnostic(t *testing.T) {
	const n = 1 << 20
	const L = uint64(128)
	const eps = 1e-3
	const gap = uint64(1) << 30

	keys := make([]uint64, n)
	for i := range keys {
		keys[i] = uint64(i) * gap
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })

	workloads := []struct {
		name    string
		weights querygen.SmartMixWeights
	}{
		{"pure_in_gap", querygen.SmartMixWeights{NearKey: 0.0, InGap: 1.0, Uniform: 0.0}},
		{"smart_mix", querygen.SmartMixWeights{NearKey: 0.5, InGap: 0.3, Uniform: 0.2}},
		{"pure_near_key", querygen.SmartMixWeights{NearKey: 1.0, InGap: 0.0, Uniform: 0.0}},
		{"pure_uniform", querygen.SmartMixWeights{NearKey: 0.0, InGap: 0.0, Uniform: 1.0}},
	}

	policies := []struct {
		name string
		p    hybridutil.FallbackPolicy
	}{
		{"AlwaysSODA", hybridutil.FallbackAlwaysSODA{}},
		{"AlwaysTrunc", hybridutil.FallbackAlwaysTrunc{}},
		{"InGapFPR", hybridutil.FallbackInGapFPR{Epsilon: eps}},
	}

	fmt.Println()
	fmt.Printf("=== Diagnostic: SegARE policies × query mixes (n=%d, L=%d, ε=%g) ===\n", n, L, eps)
	fmt.Printf("%-16s | %-12s | %10s | %12s\n", "workload", "policy", "BPK", "FPR")
	fmt.Println("---------------------------------------------------------------")

	for _, w := range workloads {
		rng := rand.New(rand.NewSource(42))
		queries := querygen.GenerateSmartQueriesWeighted(keys, 100_000, L, w.weights, rng)
		if len(queries) == 0 {
			t.Logf("workload %s: no queries generated", w.name)
			continue
		}
		for _, pol := range policies {
			r := runSegOnce(t, keys, queries, L, eps, pol.p, pol.name)
			fmt.Printf("%-16s | %-12s | %10.2f | %12.3e\n", w.name, pol.name, r.bpk, r.fpr)
		}
		fmt.Println()
	}
}
