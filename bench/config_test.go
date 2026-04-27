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
// Mirrors the entries currently in comparison_test.go's allSeries map.
var DefaultSeriesStyles = map[string]SeriesStyle{
	"Theoretical":  {Name: "Theoretical", Color: "#ef4444", Dashed: true, Marker: "circle"},
	"Grafite":      {Name: "Grafite", Color: "#1a6b3c", Marker: "diamond"},
	"SNARF":        {Name: "SNARF", Color: "#1a3a6b", Marker: "star"},
	"SuRFReal(8)":  {Name: "SuRFReal(8)", Color: "#111111", Marker: "diamond"},
	"SODA":         {Name: "SODA", Color: "#4dd88a", Marker: "diamond"},
	"Scan-ARE":     {Name: "Scan-ARE", Color: "#06b6d4", Marker: "star"},
	"Greedy+Merge": {Name: "Greedy+Merge", Color: "#22c55e", Marker: "diamond"},
	"BloomARE":     {Name: "BloomARE", Color: "#888888", Dashed: true, Marker: "circle"},
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
