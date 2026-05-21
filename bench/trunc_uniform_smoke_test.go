//go:build heavy

package bench_test

import (
	"Thesis/emptiness/approx/are_trunc"
	"fmt"
	"math/rand"
	"sort"
	"testing"
)

// TestTrunc_UniformQuery_Smoke runs Truncation on the synthetic uniform
// 16M-key set, subset to 2^20 keys, under two query strategies:
//
//  1. uniform-random across [minK, maxK] (the README's stated precondition)
//  2. smart_mix (50% near-key + 30% in-gap + 20% uniform; what B6 currently uses)
//
// Hypothesis: under (1) Truncation hits ε ≈ n·L/2^K (theoretical); under (2)
// it plateaus at ≈ 0.5 because near-key offsets ±5L sit inside the phantom
// 2^(W-K). If (1) confirms, the B6 plot is workload-induced and not a
// filter defect.
func TestTrunc_UniformQuery_Smoke(t *testing.T) {
	const (
		n          = 1 << 20
		queryCount = 1 << 18
		L          = uint64(128)
		seed       = int64(20260501)
	)

	keysAll, err := loadSOSDUint64(syntheticDataPath("uniform_16M_uint64"), 0)
	if err != nil {
		t.Skipf("uniform_16M not available: %v", err)
	}
	if len(keysAll) < n {
		t.Skipf("need %d keys, have %d", n, len(keysAll))
	}
	keys := append([]uint64(nil), keysAll[:n]...)
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })

	// Detect actual key bit-width from the data (not the file format's
	// nominal 60). Uniform synthetic keeps the full 60-bit span; we verify.
	var maxKey uint64
	for _, k := range keys {
		if k > maxKey {
			maxKey = k
		}
	}
	keyBits := uint32(64)
	for keyBits > 1 && (maxKey>>(keyBits-1)) == 0 {
		keyBits--
	}
	t.Logf("loaded %d uniform keys, maxKey=%d, keyBits=%d, span=%d", n, maxKey, keyBits, keys[n-1]-keys[0])

	rngU := rand.New(rand.NewSource(seed))
	uniformQs := generateRangeQueries(keys, queryCount, L, rngU)

	rngS := rand.New(rand.NewSource(seed))
	smartQs := generateSmartQueries(keys, queryCount, L, rngS)

	// FPR ground truth: a query [a, a+L-1] is "truly empty" if no key in
	// [a, a+L-1]. For uniform 60-bit keys with n=2^20 and L=128,
	// P(empty) is essentially 1 (n·L/2^60 ≈ 1.3e-10). So FPR ≈ filter's
	// non-empty rate.
	isTrulyEmpty := func(a, b uint64) bool {
		i := sort.Search(len(keys), func(i int) bool { return keys[i] >= a })
		return i >= len(keys) || keys[i] > b
	}

	measureFPR := func(filter *are_trunc.TruncARE, qs [][2]uint64) (fpr float64, falsePos, truEmpty int) {
		for _, q := range qs {
			a, b := q[0], q[1]
			if !isTrulyEmpty(a, b) {
				continue
			}
			truEmpty++
			if !filter.IsEmpty(a, b) {
				falsePos++
			}
		}
		if truEmpty == 0 {
			return 0, 0, 0
		}
		return float64(falsePos) / float64(truEmpty), falsePos, truEmpty
	}

	for _, K := range []uint32{36, 44, 48} {
		filter, err := are_trunc.NewTruncAREFromK(keys, keyBits, K)
		if err != nil {
			t.Fatalf("K=%d build: %v", K, err)
		}

		fU, fpU, teU := measureFPR(filter, uniformQs)
		fS, fpS, teS := measureFPR(filter, smartQs)
		bpk := float64(filter.SizeInBits()) / float64(n)
		t.Logf("K=%2d  bpk=%5.2f  uniform-q FPR=%-9s (%d/%d)  smart_mix FPR=%-9s (%d/%d)",
			K, bpk,
			fmt.Sprintf("%.4g", fU), fpU, teU,
			fmt.Sprintf("%.4g", fS), fpS, teS,
		)
	}
}
