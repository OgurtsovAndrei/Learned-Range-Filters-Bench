package sosd_test

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"Thesis/emptiness/approx/hybrid/are_seg"
	"Thesis/emptiness/approx/hybrid/hybridutil"
	"Thesis/emptiness/exact"

	"Thesis-bench-industry/bench/internal/querygen"
)

// withinAbs reports whether |a-b| <= tol.
func withinAbs(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// withinRel reports whether |a-b| <= max(absTol, relTol*max(|a|,|b|)).
func withinRel(a, b, absTol, relTol float64) bool {
	scale := math.Max(math.Abs(a), math.Abs(b))
	return math.Abs(a-b) <= math.Max(absTol, relTol*scale)
}

type segRunResult struct {
	policy    string
	bpk       float64
	fpr       float64
	nClusters int
	nFallback int
}

func runSegOnce(t *testing.T, keys []uint64, queries [][2]uint64, L uint64, eps float64, policy hybridutil.FallbackPolicy, policyName string) segRunResult {
	t.Helper()
	n := len(keys)
	K := uint32(math.Ceil(math.Log2(float64(n) * (float64(L) + 1) / eps)))
	if K == 0 {
		K = 1
	}
	if K > 64 {
		K = 64
	}
	filter, err := are_seg.NewSegAREFromKWithPolicy(keys, 64, K, L, policy, exact.VariantAuto)
	if err != nil {
		t.Fatalf("SegARE build (%s): %v", policyName, err)
	}
	fp := 0
	for _, q := range queries {
		if !filter.IsEmpty(q[0], q[1]) {
			fp++
		}
	}
	fpr := float64(fp) / float64(len(queries))
	bpk := float64(filter.SizeInBits()) / float64(len(keys))
	nC, nF, _ := filter.Stats()
	return segRunResult{
		policy:    policyName,
		bpk:       bpk,
		fpr:       fpr,
		nClusters: nC,
		nFallback: nF,
	}
}

func TestSegFallbackPolicy_Smoke_Uniform(t *testing.T) {
	const n = 1 << 20
	const L = uint64(128)
	const eps = 1e-3
	const gap = uint64(1) << 30 // truly uniform → InGapFPR should choose Trunc

	keys := make([]uint64, n)
	for i := range keys {
		keys[i] = uint64(i) * gap
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })

	rng := rand.New(rand.NewSource(42))
	queries := querygen.GenerateSmartQueriesWeighted(keys, 100_000, L,
		querygen.SmartMixWeights{
			NearKey: querygen.QueryWeightNearKey,
			InGap:   querygen.QueryWeightInGap,
			Uniform: querygen.QueryWeightUniform,
		}, rng)

	soda := runSegOnce(t, keys, queries, L, eps, hybridutil.FallbackAlwaysSODA{}, "AlwaysSODA")
	trunc := runSegOnce(t, keys, queries, L, eps, hybridutil.FallbackAlwaysTrunc{}, "AlwaysTrunc")
	inGap := runSegOnce(t, keys, queries, L, eps, hybridutil.FallbackInGapFPR{Epsilon: eps}, "InGapFPR")

	t.Logf("uniform ε=%g: SODA=(BPK=%.2f, FPR=%.3e)  Trunc=(BPK=%.2f, FPR=%.3e)  InGapFPR=(BPK=%.2f, FPR=%.3e)",
		eps, soda.bpk, soda.fpr, trunc.bpk, trunc.fpr, inGap.bpk, inGap.fpr)

	persistRow(t, "uniform", eps, soda, trunc, inGap)

	if inGap.nFallback == 0 {
		t.Logf("empty fallback — policy not exercised, skipping envelope check")
		return
	}
	sodaMatch := withinAbs(inGap.bpk, soda.bpk, 1.0) && withinRel(inGap.fpr, soda.fpr, 5e-4, 0.30)
	truncMatch := withinAbs(inGap.bpk, trunc.bpk, 1.0) && withinRel(inGap.fpr, trunc.fpr, 5e-4, 0.30)
	if !sodaMatch && !truncMatch {
		t.Errorf("InGapFPR does not match either reference line (BPK=%.2f FPR=%.3e) vs SODA (%.2f, %.3e) Trunc (%.2f, %.3e)",
			inGap.bpk, inGap.fpr, soda.bpk, soda.fpr, trunc.bpk, trunc.fpr)
	}
	if !truncMatch {
		t.Errorf("uniform: expected InGapFPR ≈ Trunc envelope (lower BPK), got Trunc=(%.2f,%.3e) vs InGapFPR=(%.2f,%.3e)",
			trunc.bpk, trunc.fpr, inGap.bpk, inGap.fpr)
	}
}

func persistRow(t *testing.T, dist string, eps float64, soda, trunc, inGap segRunResult) {
	t.Helper()
	outDir := filepath.Join("..", "..", "..", "bench_results", "data")
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		t.Logf("mkdir %s: %v", outDir, err)
		return
	}
	path := filepath.Join(outDir, "seg_fallback_policy.csv")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		t.Logf("open csv: %v", err)
		return
	}
	defer f.Close()
	fmt.Fprintf(f, "%s,%g,%s,%.4f,%.6g,%d,%d\n",
		dist, eps, soda.policy, soda.bpk, soda.fpr, soda.nClusters, soda.nFallback)
	fmt.Fprintf(f, "%s,%g,%s,%.4f,%.6g,%d,%d\n",
		dist, eps, trunc.policy, trunc.bpk, trunc.fpr, trunc.nClusters, trunc.nFallback)
	fmt.Fprintf(f, "%s,%g,%s,%.4f,%.6g,%d,%d\n",
		dist, eps, inGap.policy, inGap.bpk, inGap.fpr, inGap.nClusters, inGap.nFallback)
}
