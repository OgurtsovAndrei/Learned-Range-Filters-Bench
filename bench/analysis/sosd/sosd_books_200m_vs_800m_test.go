//go:build heavy

package sosd_test

import (
	"fmt"
	"math"
	"testing"

	"Thesis-bench-industry/bench/internal/keygen"
)

// TestBooksScalingHypothesis checks whether books_800M is a finer-resolution
// version of books_200M — e.g.\ the same underlying distribution scaled by a
// constant (seconds → nanoseconds, sales rank → raw count, etc.).
//
// The test compares the two distributions at matched percentile positions
// after sort+dedup and reports the per-percentile ratio
// books_800M.Quantile(p) / books_200M.Quantile(p).
// If the ratios are roughly constant across percentiles, the two datasets are
// related by a monotone scaling.  If they diverge, the distributions differ
// in shape, not just in scale.
func TestBooksScalingHypothesis(t *testing.T) {
	books200, err := keygen.LoadSOSDUint32(
		keygen.SOSDPath("books_200M_uint32"), 0)
	if err != nil {
		t.Skipf("books_200M_uint32 not available: %v", err)
	}
	books800, err := keygen.LoadSOSDUint64(
		keygen.SOSDPath("books_800M_uint64"), 0)
	if err != nil {
		t.Skipf("books_800M_uint64 not available: %v", err)
	}

	// keygen helpers already return sorted, but be explicit about dedup
	books200 = dedupSorted(books200)
	books800 = dedupSorted(books800)

	fmt.Printf("\n=== books_200M vs books_800M — scaling hypothesis ===\n")
	fmt.Printf("books_200M: n=%d, min=%d, max=%d\n",
		len(books200), books200[0], books200[len(books200)-1])
	fmt.Printf("books_800M: n=%d, min=%d, max=%d\n",
		len(books800), books800[0], books800[len(books800)-1])
	fmt.Printf("cardinality ratio (800M/200M) = %.3f\n",
		float64(len(books800))/float64(len(books200)))
	fmt.Printf("max ratio (800M/200M)         = %.3e (~ 2^%.2f)\n",
		float64(books800[len(books800)-1])/float64(books200[len(books200)-1]),
		math.Log2(float64(books800[len(books800)-1])/float64(books200[len(books200)-1])))

	percentiles := []float64{0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999}
	fmt.Printf("\n%-8s | %18s | %18s | %12s | log2\n",
		"p", "Q_200M(p)", "Q_800M(p)", "ratio (800/200)")
	fmt.Println("-------------------------------------------------------------------------------")
	for _, p := range percentiles {
		q200 := quantile(books200, p)
		q800 := quantile(books800, p)
		var ratio float64
		if q200 > 0 {
			ratio = float64(q800) / float64(q200)
		}
		fmt.Printf("%-8.4f | %18d | %18d | %12.3e | %5.2f\n",
			p, q200, q800, ratio, math.Log2(ratio))
	}
}

func dedupSorted(xs []uint64) []uint64 {
	if len(xs) < 2 {
		return xs
	}
	j := 0
	for i := 1; i < len(xs); i++ {
		if xs[i] != xs[j] {
			j++
			xs[j] = xs[i]
		}
	}
	return xs[:j+1]
}

func quantile(sorted []uint64, p float64) uint64 {
	idx := int(p * float64(len(sorted)-1))
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}
