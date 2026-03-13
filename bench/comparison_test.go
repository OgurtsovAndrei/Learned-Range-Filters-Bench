package bench_test

import (
	"Thesis-bench-industry/grafite"
	"Thesis-bench-industry/snarf"
	"Thesis-bench-industry/surf"
	"Thesis/bits"
	"Thesis/emptiness/are"
	"Thesis/emptiness/are_bloom"
	"Thesis/emptiness/are_hybrid"
	"Thesis/emptiness/are_optimized"
	"Thesis/emptiness/are_pgm"
	"Thesis/emptiness/are_soda_hash"
	"Thesis/testutils"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"testing"
)

const mask60 = (uint64(1) << 60) - 1

func mask60Keys(raw []uint64) []uint64 {
	seen := make(map[uint64]bool, len(raw))
	masked := make([]uint64, 0, len(raw))
	for _, k := range raw {
		k &= mask60
		if !seen[k] {
			seen[k] = true
			masked = append(masked, k)
		}
	}
	sort.Slice(masked, func(i, j int) bool { return masked[i] < masked[j] })
	return masked
}

func mask60Queries(queries [][2]uint64) [][2]uint64 {
	out := make([][2]uint64, len(queries))
	for i, q := range queries {
		a := q[0] & mask60
		b := q[1] & mask60
		if b < a {
			// rangeLen crossed the mask boundary; just use a single-point query
			b = a
		}
		out[i] = [2]uint64{a, b}
	}
	return out
}

func TestTradeoff_Industry(t *testing.T) {
	rangeLens := []uint64{1, 16, 128, 1024}
	const (
		n          = 1 << 16
		queryCount = 200_000
		nClusters  = 5
		unifFrac   = 0.15
	)

	bpkSweep := []float64{4, 6, 8, 10, 12, 14, 16, 18, 20}
	epsilons := []float64{0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001}

	rng := rand.New(rand.NewSource(99))
	rawKeys, clusters := testutils.GenerateClusterDistribution(n, nClusters, unifFrac, rng)
	keys := mask60Keys(rawKeys)
	keysBS := make([]bits.BitString, len(keys))
	for i, v := range keys {
		keysBS[i] = testutils.TrieBS(v)
	}

	os.MkdirAll("../bench_results/plots", 0755)

	for _, rangeLen := range rangeLens {
		t.Run(fmt.Sprintf("L=%d", rangeLen), func(t *testing.T) {
			qrng := rand.New(rand.NewSource(12345))
			rawQueries := testutils.GenerateClusterQueries(queryCount, clusters, unifFrac, rangeLen, qrng)
			queries := mask60Queries(rawQueries)

			// ---- series map ----
			allSeries := map[string]*testutils.SeriesData{
				"Theoretical":    {Name: "Theoretical", Color: "#ef4444", Dashed: true, Marker: "circle"},
				"Grafite":        {Name: "Grafite", Color: "#22a06b", Marker: "diamond"},
				"SNARF":          {Name: "SNARF", Color: "#9b59b6", Marker: "star"},
				"SuRF":           {Name: "SuRF", Color: "#1a9cdb", Marker: "square"},
				"SuRFHash(8)":    {Name: "SuRFHash(8)", Color: "#0e6ea8", Marker: "square"},
				"SuRFReal(8)":    {Name: "SuRFReal(8)", Color: "#084d76", Marker: "square"},
				"Adaptive (t=0)": {Name: "Adaptive (t=0)", Color: "#2a7fff", Marker: "square"},
				"SODA":           {Name: "SODA", Color: "#3aa06b", Marker: "diamond"},
				"Hybrid":         {Name: "Hybrid", Color: "#c0392b", Marker: "star"},
				"CDF-ARE":        {Name: "CDF-ARE", Color: "#e05d10", Marker: "circle"},
				"BloomARE":       {Name: "BloomARE", Color: "#888888", Dashed: true, Marker: "circle"},
			}

			fmt.Printf("\n=== Industry Comparison (60-bit keys, %d keys, L=%d) ===\n", len(keys), rangeLen)
			fmt.Printf("%-16s | %8s | %14s\n", "Series", "BPK", "FPR(skip)")
			fmt.Println(strings.Repeat("-", 45))

			// ---- Theoretical ----
			for _, eps := range epsilons {
				thBPK := math.Log2(float64(rangeLen) / eps)
				allSeries["Theoretical"].Points = append(allSeries["Theoretical"].Points,
					testutils.Point{X: thBPK, Y: eps})
			}

			// ---- ARE filters (parameterised by epsilon) ----
			for _, eps := range epsilons {
				thBPK := math.Log2(float64(rangeLen) / eps)

				// Adaptive (t=0)
				fOpt, errOpt := are_optimized.NewOptimizedARE(keysBS, rangeLen, eps, 0)
				if errOpt == nil {
					bpk := float64(fOpt.SizeInBits()) / float64(len(keys))
					fpr := testutils.MeasureFPR(keys, queries, func(a, b uint64) bool {
						return fOpt.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b))
					})
					allSeries["Adaptive (t=0)"].Points = append(allSeries["Adaptive (t=0)"].Points,
						testutils.Point{X: bpk, Y: fpr})
					fmt.Printf("%-16s | %8.2f | %14.6f\n", "Adaptive(t=0)", bpk, fpr)
				}

				// SODA
				fSoda, errSoda := are_soda_hash.NewApproximateRangeEmptinessSoda(keys, rangeLen, eps)
				if errSoda == nil {
					bpk := float64(fSoda.SizeInBits()) / float64(len(keys))
					fpr := testutils.MeasureFPR(keys, queries, func(a, b uint64) bool {
						return fSoda.IsEmpty(a, b)
					})
					allSeries["SODA"].Points = append(allSeries["SODA"].Points,
						testutils.Point{X: bpk, Y: fpr})
					fmt.Printf("%-16s | %8.2f | %14.6f\n", "SODA", bpk, fpr)
				}

				// Hybrid
				fHybrid, errHybrid := are_hybrid.NewHybridARE(keysBS, rangeLen, eps)
				if errHybrid == nil {
					bpk := float64(fHybrid.SizeInBits()) / float64(len(keys))
					fpr := testutils.MeasureFPR(keys, queries, func(a, b uint64) bool {
						return fHybrid.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b))
					})
					allSeries["Hybrid"].Points = append(allSeries["Hybrid"].Points,
						testutils.Point{X: bpk, Y: fpr})
					fmt.Printf("%-16s | %8.2f | %14.6f\n", "Hybrid", bpk, fpr)
				}

				// CDF-ARE
				fCdf, errCdf := are_pgm.NewPGMApproximateRangeEmptiness(keys, rangeLen, eps, 64)
				if errCdf == nil {
					bpk := float64(fCdf.TotalSizeInBits()) / float64(len(keys))
					fpr := testutils.MeasureFPR(keys, queries, func(a, b uint64) bool {
						return fCdf.IsEmpty(a, b)
					})
					allSeries["CDF-ARE"].Points = append(allSeries["CDF-ARE"].Points,
						testutils.Point{X: bpk, Y: fpr})
					fmt.Printf("%-16s | %8.2f | %14.6f\n", "CDF-ARE", bpk, fpr)
				}

				// BloomARE
				fBloom, errBloom := are_bloom.NewBloomARE(keys, rangeLen, eps)
				if errBloom == nil {
					bpk := float64(fBloom.SizeInBits()) / float64(len(keys))
					fpr := testutils.MeasureFPR(keys, queries, func(a, b uint64) bool {
						return fBloom.IsEmpty(a, b)
					})
					allSeries["BloomARE"].Points = append(allSeries["BloomARE"].Points,
						testutils.Point{X: bpk, Y: fpr})
					fmt.Printf("%-16s | %8.2f | %14.6f\n", "BloomARE", bpk, fpr)
				}

				_ = thBPK
			}

			// ---- Industry filters (parameterised by BPK sweep) ----
			for _, bpk := range bpkSweep {
				// Grafite
				fGrafite := grafite.New(keys, bpk)
				actualBPK := float64(fGrafite.SizeInBits()) / float64(len(keys))
				fpr := testutils.MeasureFPR(keys, queries, func(a, b uint64) bool {
					return fGrafite.IsEmpty(a, b)
				})
				allSeries["Grafite"].Points = append(allSeries["Grafite"].Points,
					testutils.Point{X: actualBPK, Y: fpr})
				fmt.Printf("%-16s | %8.2f | %14.6f\n", fmt.Sprintf("Grafite(bpk=%.0f)", bpk), actualBPK, fpr)

				// SNARF
				fSnarf := snarf.New(keys, bpk)
				actualBPKSnarf := float64(fSnarf.SizeInBits()) / float64(len(keys))
				fprSnarf := testutils.MeasureFPR(keys, queries, func(a, b uint64) bool {
					return fSnarf.IsEmpty(a, b)
				})
				allSeries["SNARF"].Points = append(allSeries["SNARF"].Points,
					testutils.Point{X: actualBPKSnarf, Y: fprSnarf})
				fmt.Printf("%-16s | %8.2f | %14.6f\n", fmt.Sprintf("SNARF(bpk=%.0f)", bpk), actualBPKSnarf, fprSnarf)
			}

			// ---- SuRF variants (single point each) ----
			type surfVariant struct {
				name     string
				st       surf.SuffixType
				hashBits int
				realBits int
			}
			surfVariants := []surfVariant{
				{"SuRF", surf.SuffixNone, 0, 0},
				{"SuRFHash(8)", surf.SuffixHash, 8, 0},
				{"SuRFReal(8)", surf.SuffixReal, 0, 8},
			}
			for _, sv := range surfVariants {
				fSurf := surf.New(keys, sv.st, sv.hashBits, sv.realBits)
				actualBPK := float64(fSurf.SizeInBits()) / float64(len(keys))
				fpr := testutils.MeasureFPR(keys, queries, func(a, b uint64) bool {
					return fSurf.IsEmpty(a, b)
				})
				allSeries[sv.name].Points = append(allSeries[sv.name].Points,
					testutils.Point{X: actualBPK, Y: fpr})
				fmt.Printf("%-16s | %8.2f | %14.6f\n", sv.name, actualBPK, fpr)
			}

			// ---- Generate SVG ----
			orderedSeries := []testutils.SeriesData{
				*allSeries["Theoretical"],
				*allSeries["Grafite"],
				*allSeries["SNARF"],
				*allSeries["SuRF"],
				*allSeries["SuRFHash(8)"],
				*allSeries["SuRFReal(8)"],
				*allSeries["Adaptive (t=0)"],
				*allSeries["SODA"],
				*allSeries["Hybrid"],
				*allSeries["CDF-ARE"],
				*allSeries["BloomARE"],
			}

			svgPath := fmt.Sprintf("../bench_results/plots/industry_cluster_L%d.svg", rangeLen)
			err := testutils.GenerateTradeoffSVG(
				fmt.Sprintf("FPR vs BPK — Industry+ARE Comparison (60-bit keys, n=%d, L=%d)", len(keys), rangeLen),
				"Bits per Key (BPK)",
				"False Positive Rate (FPR)",
				orderedSeries,
				svgPath,
			)
			if err != nil {
				t.Errorf("SVG generation failed: %v", err)
			} else {
				fmt.Printf("\nSVG written to %s\n", svgPath)
			}
		})
	}
}

// TestSanity_Grafite verifies basic Grafite behaviour.
// Filters may have false positives, so we only verify no false negatives and size > 0.
// Keys must be spread across a large enough universe for the requested bpk:
// required reduced universe r = ceil(n * 2^(bpk-2)) must be <= actual universe size.
func TestSanity_Grafite(t *testing.T) {
	keys := []uint64{0, 1_000_000_000, 2_000_000_000}
	f := grafite.New(keys, 6.0)
	if f.SizeInBits() == 0 {
		t.Error("expected SizeInBits > 0")
	}
	// No false negatives: a range containing a key must NOT be reported empty
	if f.IsEmpty(0, 1) {
		t.Error("false negative: IsEmpty(0,1) must be false — key 0 is in range")
	}
	if f.IsEmpty(999_999_999, 1_000_000_001) {
		t.Error("false negative: key 1e9 is in range")
	}
}

// TestSanity_SuRF verifies basic SuRF behaviour (no false negatives, size > 0).
func TestSanity_SuRF(t *testing.T) {
	keys := []uint64{10, 20, 30}
	f := surf.New(keys, surf.SuffixNone, 0, 0)
	if f.SizeInBits() == 0 {
		t.Error("expected SizeInBits > 0")
	}
	if f.IsEmpty(9, 11) {
		t.Error("false negative: key 10 is in range [9,11]")
	}
	if f.IsEmpty(19, 21) {
		t.Error("false negative: key 20 is in range [19,21]")
	}
}

// TestSanity_SNARF verifies basic SNARF behaviour (no false negatives, size > 0).
func TestSanity_SNARF(t *testing.T) {
	keys := []uint64{0, 1_000_000_000, 2_000_000_000}
	f := snarf.New(keys, 6.0)
	if f.SizeInBits() == 0 {
		t.Error("expected SizeInBits > 0")
	}
	if f.IsEmpty(0, 1) {
		t.Error("false negative: key 0 is in range [0,1]")
	}
	if f.IsEmpty(999_999_999, 1_000_000_001) {
		t.Error("false negative: key 1e9 is in range")
	}
}

// Truncation filter is imported to satisfy unused-import check in Go's strict build
var _ = are.NewApproximateRangeEmptiness
