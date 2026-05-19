package bench_test

import (
	"Thesis-bench-industry/bench/internal/benchutil"
	"Thesis-bench-industry/bench/internal/keygen"
	"Thesis-bench-industry/bench/internal/querygen"
	"fmt"
	"math/rand"
)

// ---- Global parameters (single source of truth across all bench tests) ----

const (
	DefaultNRuns      = 3
	DefaultQueryCount = 1 << 18 // 262144
	DefaultXMax       = 25.0    // hard cap on FPR-vs-BPK plot X-axis
	Mask60            = keygen.Mask60
)

// ---- Helpers for bench_test package ----

func sosdPath(name string) string {
	return keygen.SOSDPath(name)
}

func loadSOSDUint64(path string, maxKeys int) ([]uint64, error) {
	return keygen.LoadSOSDUint64(path, maxKeys)
}

func loadSOSDUint32(path string, maxKeys int) ([]uint64, error) {
	return keygen.LoadSOSDUint32(path, maxKeys)
}

func loadFacebookKeys(n int) ([]uint64, error) {
	return keygen.LoadSOSDUint64(keygen.SOSDPath("fb_200M_uint64"), n)
}

func loadWikiKeys(n int) ([]uint64, error) {
	return keygen.LoadSOSDUint64(keygen.SOSDPath("wiki_ts_200M_uint64"), n)
}

func loadOSMKeys(n int) ([]uint64, error) {
	return keygen.LoadSOSDUint64(keygen.SOSDPath("osm_cellids_800M_uint64"), n)
}

func loadBooksKeys(n int) ([]uint64, error) {
	return keygen.LoadSOSDUint32(keygen.SOSDPath("books_200M_uint32"), n)
}

func loadBooks800MKeys(n int) ([]uint64, error) {
	return keygen.LoadSOSDUint64(keygen.SOSDPath("books_800M_uint64"), n)
}

func generateSmartQueries(keys []uint64, count int, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	return querygen.GenerateSmartQueries(keys, count, rangeLen, rng)
}

func generateSmartQueriesWeighted(keys []uint64, count int, rangeLen uint64, w querygen.SmartMixWeights, rng *rand.Rand) [][2]uint64 {
	return querygen.GenerateSmartQueriesWeighted(keys, count, rangeLen, w, rng)
}

func generateMixedQueriesWeighted(keys []uint64, count int, rangeLen uint64, w querygen.SmartMixWeights, rng *rand.Rand) [][2]uint64 {
	return querygen.GenerateMixedQueriesWeighted(keys, count, rangeLen, w, rng)
}

func generateRangeQueries(keys []uint64, count int, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	return querygen.GenerateRangeQueries(keys, count, rangeLen, rng)
}

type smartMixWeights = querygen.SmartMixWeights

var defaultSmartMix = querygen.DefaultSmartMix

const (
	queryWeightNearKey = querygen.QueryWeightNearKey
	queryWeightInGap   = querygen.QueryWeightInGap
	queryWeightUniform = querygen.QueryWeightUniform
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

	// Epsilon sweep for the L-aware Grafite-tuned series. One point per eps,
	// with a fresh filter built per (distribution, L) pair. Allows Grafite to
	// access BPK regimes the bpk-only build cannot reach (since the library
	// caps bpk-only builds at log2(U/n) + 2).
	DefaultGrafiteEpsSweep = []float64{0.5, 0.1, 0.05, 0.01, 0.001, 1e-4, 1e-5}

	// Epsilon points for BloomARE family (canonical 10-point set).
	DefaultEpsilons = []float64{0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001}

	// 7-point variant used in performance_test.go (TestTradeoff_Full).
	EpsilonsVariant = []float64{0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001}
)

// SeriesStyle describes plot appearance for a single series.
type SeriesStyle = benchutil.SeriesStyle

// DefaultSeriesStyles is the unified 8-series set used on FPR-vs-BPK plots.
var DefaultSeriesStyles = benchutil.DefaultSeriesStyles

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
//
//	ΔBPK ≥ AdaptiveBPKGap AND
//	|Δlog10(FPR)| ≥ AdaptiveLogFPRDrop
//
// We also extend the tail with a single +AdaptiveTailStep BPK probe while the
// last measured FPR is still above the noise floor and BPK < DefaultXMax.
const (
	AdaptiveBPKGap     = 2.0 // BPK units
	AdaptiveLogFPRDrop = 1.5 // log10 units
	AdaptiveTailStep   = 2.0 // BPK units appended per tail extension
)
