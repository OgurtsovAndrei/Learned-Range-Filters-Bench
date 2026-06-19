package paramselect

import (
	"math"
	"math/rand"
	"testing"
)

func TestWidthSelectionHelpers(t *testing.T) {
	if got := WidthForBitLength(255); got != Width8 {
		t.Fatalf("WidthForBitLength(255)=%d, want %d", got, Width8)
	}
	if got := WidthForBitLength(256); got != Width16 {
		t.Fatalf("WidthForBitLength(256)=%d, want %d", got, Width16)
	}
	if got := WidthForMaxValue(255); got != Width8 {
		t.Fatalf("WidthForMaxValue(255)=%d, want %d", got, Width8)
	}
	if got := WidthForCountWithSentinel(256); got != Width16 {
		t.Fatalf("WidthForCountWithSentinel(256)=%d, want %d", got, Width16)
	}
	if got := DelimiterTrieNodeUpperBound(100); got != 200 {
		t.Fatalf("DelimiterTrieNodeUpperBound(100)=%d, want 200", got)
	}
	if got := WidthForDelimiterTrieIndex(100); got != Width8 {
		t.Fatalf("WidthForDelimiterTrieIndex(100)=%d, want %d", got, Width8)
	}
}

func TestSignatureBitsRelativeTrie_SatisfiesTheoremBound(t *testing.T) {
	cases := []struct {
		n int
		w int
		b int
	}{
		{n: 1 << 10, w: 64, b: 256},
		{n: 1 << 13, w: 128, b: 256},
		{n: 1 << 15, w: 256, b: 256},
		{n: 1 << 18, w: 1024, b: 256},
	}

	for _, tc := range cases {
		m := BucketCount(tc.n, tc.b)
		sBits := SignatureBitsRelativeTrie(tc.w, tc.n, m)
		epsilon := float64(m) / float64(tc.n)
		checks := max(1, int(math.Ceil(math.Log2(float64(max(2, tc.w))))))
		unionBound := float64(checks) * math.Pow(2, -float64(sBits))

		if unionBound > epsilon*(1.0+1e-12) {
			t.Fatalf("bound violated for n=%d w=%d m=%d: checks*2^-S=%.6g > epsilon=%.6g (S=%d)",
				tc.n, tc.w, m, unionBound, epsilon, sBits)
		}
	}
}

func TestSignatureBitsRelativeTrie_MonteCarloVsTheory(t *testing.T) {
	cases := []struct {
		n int
		w int
		b int
	}{
		{n: 1 << 10, w: 64, b: 256},
		{n: 1 << 13, w: 128, b: 256},
		{n: 1 << 15, w: 256, b: 256},
	}

	rng := rand.New(rand.NewSource(42))
	samples := 80_000

	for _, tc := range cases {
		m := BucketCount(tc.n, tc.b)
		sBits := SignatureBitsRelativeTrie(tc.w, tc.n, m)
		checks := max(1, int(math.Ceil(math.Log2(float64(max(2, tc.w))))))

		pSingle := math.Pow(2, -float64(sBits))
		pExact := 1.0 - math.Pow(1.0-pSingle, float64(checks))
		pUnion := float64(checks) * pSingle
		pEmp := estimateAnyFalsePositiveProbability(rng, sBits, checks, samples)

		absDiff := math.Abs(pEmp - pExact)
		tol := math.Max(0.002, 0.35*pExact)
		if absDiff > tol {
			t.Fatalf("empirical mismatch for n=%d w=%d (S=%d checks=%d): empirical=%.6g exact=%.6g union=%.6g diff=%.6g tol=%.6g",
				tc.n, tc.w, sBits, checks, pEmp, pExact, pUnion, absDiff, tol)
		}
		if pEmp > pUnion+0.01 {
			t.Fatalf("empirical above union bound for n=%d w=%d (S=%d checks=%d): empirical=%.6g union=%.6g",
				tc.n, tc.w, sBits, checks, pEmp, pUnion)
		}
	}
}

func estimateAnyFalsePositiveProbability(rng *rand.Rand, sBits int, checks int, samples int) float64 {
	hits := 0
	for i := 0; i < samples; i++ {
		any := false
		for j := 0; j < checks; j++ {
			if randomSig(rng, sBits) == randomSig(rng, sBits) {
				any = true
				break
			}
		}
		if any {
			hits++
		}
	}
	return float64(hits) / float64(samples)
}

func randomSig(rng *rand.Rand, sBits int) uint64 {
	switch {
	case sBits >= 64:
		return rng.Uint64()
	default:
		mask := (uint64(1) << uint(sBits)) - 1
		return rng.Uint64() & mask
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
