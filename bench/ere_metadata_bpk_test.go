package bench_test

import (
	"Thesis/emptiness/exact/ere"
	"Thesis/emptiness/exact/ere_one_d"
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"testing"
)

// TestEREMetadataBPK measures the metadata-only BPK (bits per key) for the
// original two-bitvector ERE layout vs the one-vector variant on uniform
// 64-bit keys, excluding the packed suffix array which is identical across
// the two layouts. Reports both the logical bit count (D.Num()) and the
// actual rsdic allocation (D.AllocSize()*8, includes rank/select indices).
func TestEREMetadataBPK(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping metadata BPK measurement in -short mode")
	}

	sizes := []int{1 << 20, 1 << 24, 1 << 28}

	type row struct {
		n             int
		classicNumBPK float64
		oneDNumBPK    float64
		dNumBPK       float64
		dNumPct       float64
		classicAllBPK float64
		oneDAllBPK    float64
		dAllBPK       float64
		dAllPct       float64
	}

	rows := make([]row, 0, len(sizes))

	for _, n := range sizes {
		t.Run(fmt.Sprintf("N=2^%d", log2(n)), func(t *testing.T) {
			keys := generateUniformSortedUint64(n, int64(0xC0FFEE^n))

			eClassic, err := ere.NewExactRangeEmptiness(keys, 64)
			if err != nil {
				t.Fatalf("ere build failed: %v", err)
			}
			classicNum := eClassic.MetadataNumBits()
			classicAll := eClassic.MetadataAllocBits()
			eClassic = nil
			runtime.GC()

			eOneD, err := ere_one_d.NewExactRangeEmptiness(keys, 64)
			if err != nil {
				t.Fatalf("ere_one_d build failed: %v", err)
			}
			oneDNum := eOneD.MetadataNumBits()
			oneDAll := eOneD.MetadataAllocBits()
			eOneD = nil
			keys = nil
			runtime.GC()

			r := row{
				n:             n,
				classicNumBPK: float64(classicNum) / float64(n),
				oneDNumBPK:    float64(oneDNum) / float64(n),
				classicAllBPK: float64(classicAll) / float64(n),
				oneDAllBPK:    float64(oneDAll) / float64(n),
			}
			r.dNumBPK = r.oneDNumBPK - r.classicNumBPK
			r.dNumPct = r.dNumBPK / r.classicNumBPK * 100
			r.dAllBPK = r.oneDAllBPK - r.classicAllBPK
			r.dAllPct = r.dAllBPK / r.classicAllBPK * 100
			rows = append(rows, r)

			fmt.Printf("\n--- ERE metadata BPK, N=2^%d (=%d) ---\n", log2(n), n)
			fmt.Printf("  classic Num bits:   %d  (%.4f bpk)\n", classicNum, r.classicNumBPK)
			fmt.Printf("  one-d   Num bits:   %d  (%.4f bpk)\n", oneDNum, r.oneDNumBPK)
			fmt.Printf("  classic Alloc bits: %d  (%.4f bpk)\n", classicAll, r.classicAllBPK)
			fmt.Printf("  one-d   Alloc bits: %d  (%.4f bpk)\n", oneDAll, r.oneDAllBPK)
			fmt.Printf("  Delta Num:   %+.4f bpk (%+.2f%%)\n", r.dNumBPK, r.dNumPct)
			fmt.Printf("  Delta Alloc: %+.4f bpk (%+.2f%%)\n", r.dAllBPK, r.dAllPct)
		})
	}

	fmt.Printf("\n=== Metadata BPK summary (uniform random uint64) ===\n")
	fmt.Printf("| n     | classic Num bpk | one-d Num bpk | dNum bpk | dNum %%  | classic Alloc bpk | one-d Alloc bpk | dAlloc bpk | dAlloc %% |\n")
	fmt.Printf("|-------|----------------:|--------------:|---------:|--------:|------------------:|----------------:|-----------:|---------:|\n")
	for _, r := range rows {
		fmt.Printf("| 2^%-3d | %15.4f | %13.4f | %+8.4f | %+6.2f | %17.4f | %15.4f | %+10.4f | %+7.2f |\n",
			log2(r.n),
			r.classicNumBPK, r.oneDNumBPK, r.dNumBPK, r.dNumPct,
			r.classicAllBPK, r.oneDAllBPK, r.dAllBPK, r.dAllPct)
	}
	fmt.Println()
}

func generateUniformSortedUint64(n int, seed int64) []uint64 {
	r := rand.New(rand.NewSource(seed))
	keys := make([]uint64, n)
	for i := range keys {
		keys[i] = r.Uint64()
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys
}

func log2(n int) int {
	if n <= 0 {
		return 0
	}
	k := 0
	for n > 1 {
		n >>= 1
		k++
	}
	return k
}
