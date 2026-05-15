package bench_test

import (
	"Thesis-bench-industry/thirdparty/grafite"
	"Thesis-bench-industry/thirdparty/rosetta"
	"Thesis-bench-industry/thirdparty/snarf"
	"Thesis-bench-industry/thirdparty/surf"
	"Thesis/emptiness/approx/are_bloom"
	are_hybrid_scan "Thesis/emptiness/approx/hybrid/are_dbscan"
	are_greedy_scan "Thesis/emptiness/approx/hybrid/are_greedy"
	"Thesis/emptiness/approx/hybrid/are_seg"
	"Thesis/emptiness/approx/hybrid/hybridutil"
	"Thesis/emptiness/approx/are_soda_hash"
	exactbackend "Thesis/emptiness/exact"
	"fmt"
	"math"
)

var (
	// Standard grids.
	b6SweepK   = []float64{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 28, 32, 36, 40, 48}
	b6SweepBPK = []float64{4, 6, 8, 10, 12, 14, 16, 18, 20}

	// BloomARE is BPK-driven but FPR grows exponentially with L. At
	// L=65536 / n=2^20 we already need 16 GB at eps=0.0005. We trim the
	// smallest eps values for Bloom so peak memory stays manageable
	// (~1.6 GB worst case at n=2^20). For n=2^24 / n=2^28 the runner
	// will clip further if needed; this minimal grid is the safe floor.
	b6SweepBloomEps = []float64{0.1, 0.05, 0.02, 0.01, 0.005}
	b6SweepRealBits = []float64{0, 2, 4, 8, 12, 16}
	b6SweepHashBits = []float64{2, 4, 8, 12, 16}
	b6SweepNoneBits = []float64{0}
)

type b6FilterDef struct {
	name string
	// sweepName describes the parameter being swept ("eps", "bpk", "real_bits").
	sweepName string
	// sweepValues is the grid of values for the swept parameter.
	sweepValues []float64
	// build returns an isEmpty closure plus the actual filter footprint
	// (bits). Used for pure-Go filters and Bloom (no CGo crossings).
	// Exactly one of build / buildBatch must be set per filter.
	//
	// sampleQueries is the per-L representative query sample used by
	// L-dependent CGo filters (Rosetta) for build-time level shaping.
	// Pure-Go filters ignore it.
	build func(sweep float64, sampleQueries [][2]uint64) (isEmpty func(a, b uint64) bool, sizeBits uint64, err error)
	// buildBatch is set ONLY for CGo filters. When non-nil, the runner uses
	// it instead of looping per-query through `isEmpty`. This avoids
	// ~50–200 ns of CGo crossing overhead per query and is the only way to
	// measure CGo filter latency representatively.
	buildBatch func(sweep float64, sampleQueries [][2]uint64) (queryBatch func([][2]uint64) []bool, sizeBits uint64, err error)
	// lDependent: when true, the runner rebuilds the filter per (sweep, L)
	// rather than once per sweep. Used by Rosetta whose build accepts the
	// L-specific query sample for `calc_dst` level shaping.
	lDependent bool
	// skipDists is the set of distribution names for which this filter is
	// known to be unsafe (e.g. SuRF SIGSEGVs on sosd_wiki due to upstream
	// efficient/SuRF#8). The runner skips these cells without attempting
	// the build, preserving the rest of the sweep.
	skipDists map[string]bool
	// skipLs is the set of range lengths for which this filter is too
	// expensive to measure productively (e.g. BloomARE's IsEmpty scans L
	// hash probes per query, so L≥4096 becomes minutes/cell with no
	// useful FPR signal anyway).
	skipLs map[uint64]bool
	// numClusters is written by the build closure after each build and
	// reflects how many segments/clusters the last-built instance contains.
	// Nil for non-segmented filters (SODA, Bloom, Grafite, etc.).
	numClusters *int
}

func buildB6Filters(keys []uint64, keyBits uint32) []b6FilterDef {
	// Pre-allocate cluster-count pointers. Each is shared between the
	// b6FilterDef.numClusters field and the corresponding build closure so
	// the runner can read the count written during buildOnce.
	scanAreTruncNC := new(int)
	scanAreSODANC := new(int)
	scanAreSODAPEFNC := new(int)
	scanAreSODAFbPEFNC := new(int)
	greedyTruncNC := new(int)
	greedySODANC := new(int)
	segARENC := new(int)

	return []b6FilterDef{
		{
			name: "SODA", sweepName: "K", sweepValues: b6SweepK,
			build: func(sweep float64, _ [][2]uint64) (func(a, b uint64) bool, uint64, error) {
				// Hash seed: derive from K so different K values still
				// get different hash A/B (otherwise sweep cells would
				// share hash → not independent samples).
				f, err := are_soda_hash.NewSodaAREFromKWithBackend(keys, uint32(sweep),
					int64(sweep)*1000003+int64(len(keys)), exactbackend.VariantOneD)
				if err != nil {
					return nil, 0, err
				}
				return f.IsEmpty, f.SizeInBits(), nil
			},
		},
		{
			name: "SODA-PEF", sweepName: "K", sweepValues: b6SweepK,
			build: func(sweep float64, _ [][2]uint64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_soda_hash.NewSodaAREFromKWithBackend(keys, uint32(sweep),
					int64(sweep)*1000003+int64(len(keys)), exactbackend.VariantPEF)
				if err != nil {
					return nil, 0, err
				}
				return f.IsEmpty, f.SizeInBits(), nil
			},
		},
		{
			name: "Scan-ARE-Trunc", sweepName: "K", sweepValues: b6SweepK,
			numClusters: scanAreTruncNC,
			build: func(sweep float64, _ [][2]uint64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_hybrid_scan.NewHybridScanAREWithPolicy(keys, keyBits,
					are_hybrid_scan.ConfigWithPolicy{K: uint32(sweep), Policy: hybridutil.FallbackAlwaysTrunc{}}.
						WithEREBackend(exactbackend.VariantOneD))
				if err != nil {
					return nil, 0, err
				}
				nc, _, _ := f.Stats()
				*scanAreTruncNC = nc
				return f.IsEmpty, f.SizeInBits(), nil
			},
		},
		{
			name: "Scan-ARE-SODA", sweepName: "K", sweepValues: b6SweepK,
			numClusters: scanAreSODANC,
			build: func(sweep float64, _ [][2]uint64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_hybrid_scan.NewHybridScanAREWithPolicy(keys, keyBits,
					are_hybrid_scan.ConfigWithPolicy{K: uint32(sweep), Policy: hybridutil.FallbackAlwaysSODA{}}.
						WithEREBackend(exactbackend.VariantOneD))
				if err != nil {
					return nil, 0, err
				}
				nc, _, _ := f.Stats()
				*scanAreSODANC = nc
				return f.IsEmpty, f.SizeInBits(), nil
			},
		},
		{
			// PEF everywhere: cluster sub-filters + SODA fallback.
			name: "Scan-ARE-SODA-PEF", sweepName: "K", sweepValues: b6SweepK,
			numClusters: scanAreSODAPEFNC,
			build: func(sweep float64, _ [][2]uint64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_hybrid_scan.NewHybridScanAREWithPolicy(keys, keyBits,
					are_hybrid_scan.ConfigWithPolicy{K: uint32(sweep), Policy: hybridutil.FallbackAlwaysSODA{}}.
						WithEREBackend(exactbackend.VariantPEF))
				if err != nil {
					return nil, 0, err
				}
				nc, _, _ := f.Stats()
				*scanAreSODAPEFNC = nc
				return f.IsEmpty, f.SizeInBits(), nil
			},
		},
		{
			// PEF in SODA fallback only; cluster sub-filters use OneD.
			name: "Scan-ARE-SODA-FbPEF", sweepName: "K", sweepValues: b6SweepK,
			numClusters: scanAreSODAFbPEFNC,
			build: func(sweep float64, _ [][2]uint64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_hybrid_scan.NewHybridScanAREWithPolicy(keys, keyBits,
					are_hybrid_scan.ConfigWithPolicy{K: uint32(sweep), Policy: hybridutil.FallbackAlwaysSODA{}}.
						WithEREBackend(exactbackend.VariantOneD).
						WithFallbackEREBackend(exactbackend.VariantPEF))
				if err != nil {
					return nil, 0, err
				}
				nc, _, _ := f.Stats()
				*scanAreSODAFbPEFNC = nc
				return f.IsEmpty, f.SizeInBits(), nil
			},
		},
		{
			name: "Greedy+Merge-Trunc", sweepName: "K", sweepValues: b6SweepK,
			numClusters: greedyTruncNC,
			build: func(sweep float64, _ [][2]uint64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_greedy_scan.NewGreedyScanAREWithPolicy(keys, keyBits,
					are_greedy_scan.ConfigWithPolicy{K: uint32(sweep), Policy: hybridutil.FallbackAlwaysTrunc{}}.
						WithEREBackend(exactbackend.VariantOneD))
				if err != nil {
					return nil, 0, err
				}
				nc, _, _ := f.Stats()
				*greedyTruncNC = nc
				return f.IsEmpty, f.SizeInBits(), nil
			},
		},
		{
			name: "Greedy+Merge-SODA", sweepName: "K", sweepValues: b6SweepK,
			numClusters: greedySODANC,
			build: func(sweep float64, _ [][2]uint64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_greedy_scan.NewGreedyScanAREWithPolicy(keys, keyBits,
					are_greedy_scan.ConfigWithPolicy{K: uint32(sweep), Policy: hybridutil.FallbackAlwaysSODA{}}.
						WithEREBackend(exactbackend.VariantOneD))
				if err != nil {
					return nil, 0, err
				}
				nc, _, _ := f.Stats()
				*greedySODANC = nc
				return f.IsEmpty, f.SizeInBits(), nil
			},
		},
		{
			name: "SegARE", sweepName: "K", sweepValues: b6SweepK,
			numClusters: segARENC,
			build: func(sweep float64, _ [][2]uint64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_seg.NewSegAREFromKWithBackend(keys, keyBits, uint32(sweep), 1, exactbackend.VariantOneD)
				if err != nil {
					return nil, 0, err
				}
				nc, _, _ := f.Stats()
				*segARENC = nc
				return f.IsEmpty, f.SizeInBits(), nil
			},
		},
		{
			name:        "BloomARE",
			sweepName:   "K",
			sweepValues: b6SweepK,
			skipLs:      map[uint64]bool{4096: true, 16384: true, 65536: true},
			build: func(sweep float64, _ [][2]uint64) (func(a, b uint64) bool, uint64, error) {
				bpk := sweep
				estBits := float64(len(keys)) * bpk
				if estBits > 1.6e10 {
					return nil, 0, fmt.Errorf("bloom: estimated %.2g bits exceeds 2 GB envelope at BPK=%.1f", estBits, bpk)
				}
				pointFPR := math.Exp(-bpk * 0.4804530139182014)
				f, err := are_bloom.NewBloomAREFromPointFPR(keys, pointFPR)
				if err != nil {
					return nil, 0, err
				}
				return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), nil
			},
		},
		{
			name: "Grafite", sweepName: "bpk", sweepValues: b6SweepBPK,
			buildBatch: func(sweep float64, _ [][2]uint64) (func([][2]uint64) []bool, uint64, error) {
				f := tryGrafite(keys, sweep)
				if f == nil {
					return nil, 0, fmt.Errorf("grafite: target bpk=%.2f exceeds envelope", sweep)
				}
				return f.QueryBatch, f.SizeInBits(), nil
			},
		},
		{
			name: "SNARF", sweepName: "bpk", sweepValues: b6SweepBPK,
			buildBatch: func(sweep float64, _ [][2]uint64) (func([][2]uint64) []bool, uint64, error) {
				f := snarf.New(keys, sweep)
				return f.QueryBatch, f.SizeInBits(), nil
			},
		},
		{
			name: "Rosetta", sweepName: "bpk", sweepValues: b6SweepBPK,
			lDependent: true,
			skipLs:     map[uint64]bool{4096: true, 65536: true},
			buildBatch: func(sweep float64, sampleQueries [][2]uint64) (func([][2]uint64) []bool, uint64, error) {
				sampleN := len(sampleQueries)
				var sampleLeft, sampleRight []uint64
				if sampleN > 0 {
					sampleLeft = make([]uint64, sampleN)
					sampleRight = make([]uint64, sampleN)
					for i, q := range sampleQueries {
						sampleLeft[i] = q[0]
						sampleRight[i] = q[1]
					}
				}
				f := rosetta.New(keys, sweep, sampleLeft, sampleRight)
				if f == nil {
					return nil, 0, fmt.Errorf("rosetta: New returned nil for bpk=%.2f", sweep)
				}
				return f.QueryBatch, f.SizeInBits(), nil
			},
		},
		{
			name:        "SuRFNone",
			sweepName:   "real_bits",
			sweepValues: b6SweepNoneBits,
			buildBatch: func(_ float64, _ [][2]uint64) (func([][2]uint64) []bool, uint64, error) {
				f := surf.New(keys, surf.SuffixNone, 0, 0)
				return f.QueryBatch, f.SizeInBits(), nil
			},
			skipDists: map[string]bool{"sosd_wiki": true},
		},
		{
			name:        "SuRFHash",
			sweepName:   "hash_bits",
			sweepValues: b6SweepHashBits,
			buildBatch: func(sweep float64, _ [][2]uint64) (func([][2]uint64) []bool, uint64, error) {
				f := surf.New(keys, surf.SuffixHash, int(sweep), 0)
				return f.QueryBatch, f.SizeInBits(), nil
			},
			skipDists: map[string]bool{"sosd_wiki": true},
		},
		{
			name:        "SuRFReal",
			sweepName:   "real_bits",
			sweepValues: b6SweepRealBits,
			buildBatch: func(sweep float64, _ [][2]uint64) (func([][2]uint64) []bool, uint64, error) {
				f := surf.New(keys, surf.SuffixReal, 0, int(sweep))
				return f.QueryBatch, f.SizeInBits(), nil
			},
			skipDists: map[string]bool{"sosd_wiki": true},
		},
	}
}

// tryGrafite builds a Grafite filter at the requested bpk.
func tryGrafite(keys []uint64, bpk float64) *grafite.GrafiteFilter {
	if len(keys) < 2 {
		return nil
	}
	if keys[len(keys)-1] == keys[0] {
		return nil
	}
	return grafite.New(keys, bpk)
}
