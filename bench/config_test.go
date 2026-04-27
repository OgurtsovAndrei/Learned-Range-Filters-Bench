package bench_test

import "fmt"

// ---- Global parameters (single source of truth across all bench tests) ----

const (
	DefaultNRuns      = 3
	DefaultQueryCount = 1 << 18 // 262144
	DefaultXMax       = 25.0    // hard cap on FPR-vs-BPK plot X-axis
	Mask60            = (uint64(1) << 60) - 1
)

var (
	DefaultSeeds = []int64{12345, 54321, 99999}

	DefaultRangeLens = []uint64{1, 16, 128, 1024, 4096, 16384, 65536}

	// K-grid for fixed-K tuning across our own filters (matches comparison_test.go).
	DefaultKGrid = []uint32{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 28, 32, 36, 40, 48}

	// Extended K-grid used in hybrid_compare_test.go (DefaultKGrid + 52, 56).
	DefaultKGridExtended = []uint32{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 28, 32, 36, 40, 48, 52, 56}

	// BPK sweep for CGo filters (Grafite/SNARF/SuRFReal).
	DefaultBPKSweep = []float64{4, 6, 8, 10, 12, 14, 16, 18, 20}

	// Epsilon points for BloomARE family (canonical 10-point set).
	DefaultEpsilons = []float64{0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001}

	// 7-point variant used in performance_test.go (TestTradeoff_Full).
	EpsilonsVariant = []float64{0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001}
)

// SeriesStyle describes plot appearance for a single series.
type SeriesStyle struct {
	Name   string
	Color  string
	Marker string
	Dashed bool
}

// DefaultSeriesStyles is the unified 8-series set used on FPR-vs-BPK plots.
// Two-axis visual contrast separates the academic line (this work + its
// baseline) from the industry alternative:
//
//	                          Colour                              Marker
//	This work + Goswami SODA  WARM (mustard, orange, fuchsia)     circle
//	Industry baselines        COOL DARK (teal, navy, near-black)  diamond
//	References                MUTED DASHED (slate, gray)          circle
//
// Scan-ARE is the headline so it gets the brightest accent (fuchsia).
// SODA is the academic predecessor this work builds upon — same shape as
// Greedy+Merge / Scan-ARE, distinct (warmer) hue.
var DefaultSeriesStyles = map[string]SeriesStyle{
	"Theoretical":  {Name: "Theoretical", Color: "#374151", Dashed: true, Marker: "circle"},
	"Grafite":      {Name: "Grafite", Color: "#0f766e", Marker: "diamond"},
	"SNARF":        {Name: "SNARF", Color: "#1e3a8a", Marker: "diamond"},
	"SuRFReal(8)":  {Name: "SuRFReal(8)", Color: "#0f172a", Marker: "diamond"},
	"SODA":         {Name: "SODA", Color: "#ca8a04", Marker: "circle"},
	"Scan-ARE":     {Name: "Scan-ARE", Color: "#d946ef", Marker: "circle"},
	"Greedy+Merge": {Name: "Greedy+Merge", Color: "#ea580c", Marker: "circle"},
	"BloomARE":     {Name: "BloomARE", Color: "#9ca3af", Dashed: true, Marker: "circle"},
}

// ---- Helpers ----

// BenchResultsDataDir returns the JSON cache path for one (n, distName) cell.
func BenchResultsDataDir(n int, distName string) string {
	return fmt.Sprintf("../bench_results/data/N%d/%s", n, distName)
}

// BenchResultsPlotsDir returns the SVG output path for one (n, distName) cell.
func BenchResultsPlotsDir(n int, distName string) string {
	return fmt.Sprintf("../bench_results/plots/N%d/%s", n, distName)
}

// DefaultYFloor returns the FPR measurement noise floor for a series.
// totalSamples = queryCount * (number of independent runs averaged).
// At qc=2^18, runs=3: floor = 1/(262144*3) ≈ 1.27e-6.
func DefaultYFloor(queryCount, runs int) float64 {
	return 1.0 / float64(queryCount*runs)
}

// ---- Adaptive refinement thresholds ----
//
// After the initial DefaultBPKSweep is measured for a CGo filter, we add a
// midpoint between adjacent measurements when:
//   ΔBPK ≥ AdaptiveBPKGap AND
//   |Δlog10(FPR)| ≥ AdaptiveLogFPRDrop
//
// We also extend the tail with a single +AdaptiveTailStep BPK probe while the
// last measured FPR is still above the noise floor and BPK < DefaultXMax.
const (
	AdaptiveBPKGap     = 2.0 // BPK units
	AdaptiveLogFPRDrop = 1.5 // log10 units
	AdaptiveTailStep   = 2.0 // BPK units appended per tail extension
)
