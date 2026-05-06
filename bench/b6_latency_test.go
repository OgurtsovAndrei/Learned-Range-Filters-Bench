package bench_test

import (
	"Thesis-bench-industry/thirdparty/rosetta"
	"Thesis-bench-industry/thirdparty/snarf"
	"Thesis-bench-industry/thirdparty/surf"
	"Thesis/emptiness/approx/are_bloom"
	"Thesis/emptiness/approx/are_greedy_scan"
	"Thesis/emptiness/approx/are_hybrid_scan"
	"Thesis/emptiness/approx/are_soda_hash"
	"fmt"
	"math"
	mathbits "math/bits"
	"math/rand"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestB6IndustryLatency measures four things per (distribution, filter, L,
// sweepValue):
//   - build throughput (M keys/sec)
//   - actual filter footprint (BPK = SizeInBits / n)
//   - query latency (ns/op) on guaranteed-empty smart-mix queries
//   - empirical FPR (= count of false-positive returns / queries),
//     since smart-mix queries are guaranteed empty
//
// Each filter additionally sweeps its own tuning parameter (eps for ARE/Bloom,
// bpk for Grafite/SNARF, real_bits for SuRFReal). This produces a trajectory
// through parameter space per (distribution, filter, L) — needed for genuine
// FPR-vs-BPK curves and for cache-pressure plots (query latency vs filter
// footprint).
//
// One pass over the query batch covers both latency and FPR — no second
// query batch needed. Headline filter set at n=2^24 across L values, on
// multiple distributions.
//
// Layout:
//
//	TestB6IndustryLatency/<distribution>/<filter>
//
// Each (distribution, filter) is its own subtest so a SIGSEGV in one CGo
// wrapper does not lose data collected by previous subtests, and one filter
// can be re-run via:
//
//	go test -run "TestB6IndustryLatency/sosd_fb/Scan-ARE" ./bench/
//
// Distribution diversity is important here: SODA has a known degenerate
// regime when the input universe fits in a single super-block of size 2^K
// (K = log2(n*L/eps)). On SOSD FB (keys < 2^33) and L >= 1, K already
// exceeds the key width, so PairwiseHash(0, ...) = 0 and SODA's hashing
// reduces to identity. The ERE underneath then sees the original
// non-uniform key distribution. Synthetic uniform / spread (keys spread
// over 2^64) and SOSD osm (uint64 cell IDs in a wider universe) test the
// non-degenerate regime where SODA actually mixes keys across blocks.
//
// Output: bench_results/data/b6_latency.json (incrementally updated per
// subtest, merged across runs).
func TestB6IndustryLatency(t *testing.T) {
	t.Cleanup(closeB6ProgressLog)

	const (
		queryCount = 1 << 18
		eps        = 0.01
	)
	rangeLens := []uint64{1, 16, 128, 1024, 4096, 16384, 65536}

	nValues := parseB6N()
	mix := parseB6QueryMix()
	if mix.name != "" {
		t.Logf("B6_QUERY_MIX=%s (near=%.2f, gap=%.2f, uniform=%.2f) — cache & plots routed to *_%s",
			mix.name, mix.weights.nearKey, mix.weights.inGap, mix.weights.uniform, mix.name)
	}

	type distSpec struct {
		name string
		// load returns at least n unique uint64 keys (sorted). The closure
		// is built per-N because synthetic distributions have a fixed-size
		// generator file and SOSD distributions cap by n.
		makeLoad func(n int) func() ([]uint64, error)
	}
	distributions := []distSpec{
		{"sosd_fb", func(n int) func() ([]uint64, error) {
			return func() ([]uint64, error) {
				return loadSOSDUint64(sosdPath("fb_200M_uint64"), 2*n)
			}
		}},
		{"sosd_wiki", func(n int) func() ([]uint64, error) {
			return func() ([]uint64, error) {
				return loadSOSDUint64(sosdPath("wiki_ts_200M_uint64"), 2*n)
			}
		}},
		{"sosd_osm", func(n int) func() ([]uint64, error) {
			return func() ([]uint64, error) {
				return loadSOSDUint64(sosdPath("osm_cellids_800M_uint64"), 2*n)
			}
		}},
		{"sosd_books", func(n int) func() ([]uint64, error) {
			return func() ([]uint64, error) {
				return loadSOSDUint32(sosdPath("books_200M_uint32"), 2*n)
			}
		}},
		{"uniform", func(n int) func() ([]uint64, error) {
			return func() ([]uint64, error) {
				return loadSOSDUint64(syntheticDataPath(syntheticFile("uniform", n)), 0)
			}
		}},
		{"spread", func(n int) func() ([]uint64, error) {
			return func() ([]uint64, error) {
				return loadSOSDUint64(syntheticDataPath(syntheticFile("spread", n)), 0)
			}
		}},
		{"clustered", func(n int) func() ([]uint64, error) {
			return func() ([]uint64, error) {
				return loadSOSDUint64(syntheticDataPath(syntheticFile("clustered", n)), 0)
			}
		}},
	}

	skipSet := map[string]bool{}
	if skip := os.Getenv("SKIP_FILTERS"); skip != "" {
		for _, name := range strings.Split(skip, ",") {
			skipSet[strings.TrimSpace(name)] = true
		}
	}

	for _, n := range nValues {
		n := n
		t.Run(fmt.Sprintf("N=2^%d", mathbits.TrailingZeros(uint(n))), func(t *testing.T) {
			store := newB6Store(n, queryCount, eps, mix.queryStrategy, mix.name)
			b6Logf("\n=== B6: Build + Query latency + actual BPK + FPR, n=%d, ε=%.3f ===\n",
				n, eps)
			b6Logf("%-11s | %-14s | %-7s | %-13s | %-9s | %-13s | %-13s | %-7s | %-9s\n",
				"Distribution", "Filter", "L", "sweep", "build_ms", "build_Mkeys/s", "query_ns/op", "bpk", "fpr")

			for _, ds := range distributions {
				ds := ds
				t.Run(ds.name, func(t *testing.T) {
					allKeys, err := ds.makeLoad(n)()
					if err != nil {
						t.Skipf("load %s: %v", ds.name, err)
					}
					// If the dataset doesn't have n keys, use whatever
					// is available and report the effective n in the
					// row's NKeys field. Plots can annotate the dist.
					effN := n
					if len(allKeys) < n {
						effN = len(allKeys)
						t.Logf("%s: requested n=%d, only %d available — using effective n=%d",
							ds.name, n, len(allKeys), effN)
					}
					if effN < 1<<10 {
						t.Skipf("dataset too small even for smallest n: %d", effN)
					}
					keys := allKeys[:effN]
					keyBits := uint32(max(1, mathbits.Len64(keys[len(keys)-1])))
					t.Logf("%s: %d keys, range [%d, %d], keyBits=%d",
						ds.name, len(keys), keys[0], keys[len(keys)-1], keyBits)

					filters := buildB6Filters(keys, keyBits)
					for _, fd := range filters {
						fd := fd
						if skipSet[fd.name] {
							t.Logf("%s/%s: skipped via SKIP_FILTERS", ds.name, fd.name)
							continue
						}
						if fd.skipDists[ds.name] {
							t.Logf("%s/%s: skipped — known unsafe combination",
								ds.name, fd.name)
							continue
						}
						// CGo wrappers (Grafite/SNARF/SuRF) hold C++
						// state that is not safe for concurrent
						// IsEmpty calls. Skip them at parallelism > 1.
						if parseB6Parallelism() > 1 && isCGoFilter(fd.name) {
							t.Logf("%s/%s: skipped — CGo filter at P>1",
								ds.name, fd.name)
							continue
						}
						t.Run(fd.name, func(t *testing.T) {
							rows := runB6Filter(t, store, ds.name, fd,
								keys, rangeLens, queryCount, effN, eps, mix)
							store.update(ds.name, fd.name, rows)
							if err := store.flush(); err != nil {
								t.Errorf("flush %s: %v", store.path(), err)
							}
						})
						// Force release of all filter structures before the
						// next filter builds — without this, BloomARE at
						// L=65536/eps=0.0005 + a second large filter held alive
						// by Go's lazy GC can push peak RSS past system RAM.
						runtime.GC()
						debug.FreeOSMemory()
					}
				})
			}

			t.Logf("wrote %s", store.path())
		})
	}
}

// syntheticFile picks the smallest available synthetic-keys file that
// can serve a request for n keys. Currently we have 16M and 256M files;
// for n > 16M we use the 256M file.
func syntheticFile(dist string, n int) string {
	if n > (1 << 24) {
		return fmt.Sprintf("%s_256M_uint64", dist)
	}
	return fmt.Sprintf("%s_16M_uint64", dist)
}

// b6QueryMix names a smart-mix workload variant.
//
//   - name: short identifier used as path suffix (cache + plots).
//     Empty for the default mix so existing layouts are untouched.
//   - queryStrategy: full label written into b6Params/b6Doc.QueryStrategy
//     (cache key participates so different mixes never collide).
//   - weights: passed to generateSmartQueriesWeighted.
type b6QueryMix struct {
	name          string
	queryStrategy string
	weights       smartMixWeights
}

// b6QueryMixes is the registry of supported B6_QUERY_MIX values. Add a
// new entry here to introduce a workload variant; the runner + cache +
// plotter pick it up automatically via the suffix.
var b6QueryMixes = map[string]b6QueryMix{
	"smart_mix": {
		name:          "", // default keeps bare paths
		queryStrategy: "smart_mix_guaranteed_empty",
		weights:       defaultSmartMix,
	},
	"gap_heavy": {
		name:          "gap_heavy",
		queryStrategy: "smart_mix_gap_heavy_guaranteed_empty",
		weights:       smartMixWeights{nearKey: 0.0, inGap: 0.7, uniform: 0.3},
	},
}

// parseB6QueryMix reads B6_QUERY_MIX env var. Default "smart_mix" matches
// the historical 50/30/20 weights and writes to the original cache/plot
// directories.
func parseB6QueryMix() b6QueryMix {
	v := strings.TrimSpace(os.Getenv("B6_QUERY_MIX"))
	if v == "" {
		v = "smart_mix"
	}
	mix, ok := b6QueryMixes[v]
	if !ok {
		panic(fmt.Sprintf("B6_QUERY_MIX: unknown mix %q (known: smart_mix, gap_heavy)", v))
	}
	return mix
}

// parseB6N reads the B6_N env var as a comma-separated list of N values.
// Each value may be a decimal integer or a 2^k notation. Default 2^24.
//
//	B6_N=1048576           → [1048576]
//	B6_N=2^20,2^24,2^26    → [1048576, 16777216, 67108864]
func parseB6N() []int {
	v := os.Getenv("B6_N")
	if v == "" {
		return []int{1 << 24}
	}
	out := []int{}
	for _, tok := range strings.Split(v, ",") {
		tok = strings.TrimSpace(tok)
		if strings.HasPrefix(tok, "2^") {
			k, err := strconv.Atoi(tok[2:])
			if err != nil || k < 0 || k > 32 {
				panic(fmt.Sprintf("B6_N: bad token %q", tok))
			}
			out = append(out, 1<<k)
			continue
		}
		n, err := strconv.Atoi(tok)
		if err != nil || n < 1 {
			panic(fmt.Sprintf("B6_N: bad token %q", tok))
		}
		out = append(out, n)
	}
	return out
}

// Per-filter sweep grids. Top-level vars so they are easy to tune for
// individual reruns without touching the filter table.
var (
	// SODA and BloomARE keep their eps-based public APIs; Truncation/Scan-ARE/
	// Greedy+Merge are now K-driven (see TruncARE/HybridScanARE/GreedyScanARE
	// Config). The K grid spans approximately the same BPK range the eps grid
	// covered (BPK ≈ K under exact-mode regimes).
	b6SweepEps      = []float64{0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005}
	b6SweepK        = []float64{4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64}
	b6SweepBPK      = []float64{4, 6, 8, 10, 12, 14, 16, 18}
	// BloomARE filter size m = n*L/eps. With the full eps grid at
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
}

func buildB6Filters(keys []uint64, keyBits uint32) []b6FilterDef {
	return []b6FilterDef{
		{
			name: "SODA", sweepName: "K", sweepValues: b6SweepK,
			build: func(sweep float64, _ [][2]uint64) (func(a, b uint64) bool, uint64, error) {
				// Hash seed: derive from K so different K values still
				// get different hash A/B (otherwise sweep cells would
				// share hash → not independent samples).
				f, err := are_soda_hash.NewSodaAREFromK(keys, uint32(sweep), int64(sweep)*1000003+int64(len(keys)))
				if err != nil {
					return nil, 0, err
				}
				return f.IsEmpty, f.SizeInBits(), nil
			},
		},
		{
			name: "Scan-ARE-Trunc", sweepName: "K", sweepValues: b6SweepK,
			build: func(sweep float64, _ [][2]uint64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_hybrid_scan.NewHybridScanAREWithPolicy(keys, keyBits,
					are_hybrid_scan.ConfigWithPolicy{K: uint32(sweep), Policy: are_hybrid_scan.FallbackAlwaysTrunc{}})
				if err != nil {
					return nil, 0, err
				}
				return f.IsEmpty, f.SizeInBits(), nil
			},
		},
		{
			name: "Scan-ARE-SODA", sweepName: "K", sweepValues: b6SweepK,
			build: func(sweep float64, _ [][2]uint64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_hybrid_scan.NewHybridScanAREWithPolicy(keys, keyBits,
					are_hybrid_scan.ConfigWithPolicy{K: uint32(sweep), Policy: are_hybrid_scan.FallbackAlwaysSODA{}})
				if err != nil {
					return nil, 0, err
				}
				return f.IsEmpty, f.SizeInBits(), nil
			},
		},
		{
			name: "Greedy+Merge-Trunc", sweepName: "K", sweepValues: b6SweepK,
			build: func(sweep float64, _ [][2]uint64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_greedy_scan.NewGreedyScanAREWithPolicy(keys, keyBits,
					are_greedy_scan.ConfigWithPolicy{K: uint32(sweep), Policy: are_greedy_scan.FallbackAlwaysTrunc{}})
				if err != nil {
					return nil, 0, err
				}
				return f.IsEmpty, f.SizeInBits(), nil
			},
		},
		{
			name: "Greedy+Merge-SODA", sweepName: "K", sweepValues: b6SweepK,
			build: func(sweep float64, _ [][2]uint64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_greedy_scan.NewGreedyScanAREWithPolicy(keys, keyBits,
					are_greedy_scan.ConfigWithPolicy{K: uint32(sweep), Policy: are_greedy_scan.FallbackAlwaysSODA{}})
				if err != nil {
					return nil, 0, err
				}
				return f.IsEmpty, f.SizeInBits(), nil
			},
		},
		{
			// BloomARE is BPK-driven and L-independent: filter size is
			// fixed by target bits-per-key, queries at any L just probe
			// the same filter L times. Range FPR = 1 - (1-pointFPR)^L
			// is implicit per-query.
			//
			// IsEmpty scans every uint64 in [a, b] against the bloom
			// filter — query time grows linearly with L. At L >= 4096
			// each cell costs minutes with no additional information,
			// so we skip those L values and rely on the smaller-L
			// trajectory plus the analytical formula
			// FPR(L) = 1 - (1-pointFPR)^L for the rest.
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
				// pointFPR from target BPK at Bloom optimum:
				// pointFPR ≈ exp(-BPK * ln(2)^2).
				pointFPR := math.Exp(-bpk * 0.4804530139182014)
				f, err := are_bloom.NewBloomAREFromPointFPR(keys, pointFPR)
				if err != nil {
					return nil, 0, err
				}
				return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), nil
			},
		},
		{
			// Grafite is L-aware via its bpk parameterisation: r = n*2^(bpk-2)
			// implicitly fixes ε_l = l / 2^(bpk-2) for a query of length l.
			// At small bpk + large L, the implied ε saturates at ~1 and FPR
			// hits the ceiling, but the cells are still informative for the
			// FPR-vs-BPK trajectory plots. The previous skipLs guard was a
			// defence against the upstream `runtime_error` thrown when r
			// exceeds the lossless envelope — now handled in
			// thirdparty/grafite/cpp/wrapper.cpp via the lossless EF
			// fallback, so all (bpk, L) cells are measurable.
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
			// Rosetta is BPK-driven and uses the per-L sample query batch
			// for `calc_dst` level shaping at build time. lDependent makes
			// the runner rebuild it per (sweep, L) with the L-specific
			// sample. Other filters reuse one build across all L values.
			name: "Rosetta", sweepName: "bpk", sweepValues: b6SweepBPK,
			lDependent: true,
			// Rosetta query latency degrades sharply at L >= 4096 (Bloom-style
			// probe count grows with log L; query cost dominated by ~10
			// concurrent BF lookups per call). At L=4096+ ns/op exceeds 90 µs,
			// well outside the competitive band — skip them in latency plots.
			skipLs: map[uint64]bool{4096: true, 65536: true},
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
		// SuRF is one filter family with three structural variants. We sweep
		// each variant's bit budget so the FPR-vs-BPK plots get a SuRF point
		// cloud across (suffixType, bitCount); the plotter folds all three
		// names into a single marker-only "SuRF" series.
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

// b6ProgressLog is the package-level append-only progress logger. Lines
// are written immediately (OS write buffer flushes via newline) so a
// concurrent `tail -f bench_results/b6_progress.log` shows real-time
// per-cell progress, regardless of `go test`'s end-of-test stdout
// buffering.
var (
	b6ProgressMu  sync.Mutex
	b6ProgressLog *os.File
)

func b6Logf(format string, args ...any) {
	b6ProgressMu.Lock()
	defer b6ProgressMu.Unlock()
	if b6ProgressLog == nil {
		f, err := os.OpenFile("../bench_results/b6_progress.log",
			os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			return
		}
		b6ProgressLog = f
		fmt.Fprintf(b6ProgressLog, "\n=== b6 run start %s pid=%d ===\n",
			time.Now().Format(time.RFC3339), os.Getpid())
	}
	fmt.Fprintf(b6ProgressLog, format, args...)
}

// closeB6ProgressLog flushes and closes the progress log. Idempotent;
// safe to call multiple times.
func closeB6ProgressLog() {
	b6ProgressMu.Lock()
	defer b6ProgressMu.Unlock()
	if b6ProgressLog == nil {
		return
	}
	fmt.Fprintf(b6ProgressLog, "=== b6 run end %s pid=%d ===\n",
		time.Now().Format(time.RFC3339), os.Getpid())
	_ = b6ProgressLog.Sync()
	_ = b6ProgressLog.Close()
	b6ProgressLog = nil
}

func runB6Filter(
	t *testing.T,
	store *b6Store,
	dist string,
	fd b6FilterDef,
	keys []uint64,
	rangeLens []uint64,
	queryCount int,
	n int,
	eps float64,
	mix b6QueryMix,
) []b6Row {
	force := os.Getenv("FORCE") != ""
	parallelism := parseB6Parallelism()

	rows := make([]b6Row, 0, len(rangeLens)*len(fd.sweepValues))
	warmedUp := false

	// lSaturated tracks per-L whether we've already observed FPR=0 at
	// a previous (smaller) K-sweep value. Once an L hits FPR=0 we skip
	// querying it at higher K — the filter is monotone, so the cell
	// would just record a duplicate. The K-sweep keeps running on the
	// remaining unsaturated L values.
	lSaturated := make(map[uint64]bool)

	// Outer loop: sweep value. For non-lDependent filters we build the
	// structure once per sweep value and reuse it across all L; for
	// lDependent filters (Rosetta) we rebuild per (sweep, L) since the
	// build closure consumes an L-specific query sample.
	for _, sweep := range fd.sweepValues {
		var (
			builtIsEmpty    func(a, b uint64) bool
			builtQueryBatch func([][2]uint64) []bool
			builtSizeBits   uint64
			builtBuildDur   time.Duration
			builtBuildErr   error
			built           bool
		)
		// invokeBuild calls fd.build / fd.buildBatch and returns the
		// resulting query closures alongside size and duration. Exactly
		// one of (isEmpty, queryBatch) is non-nil on success.
		invokeBuild := func(sample [][2]uint64) (
			func(a, b uint64) bool, func([][2]uint64) []bool, uint64, time.Duration, error,
		) {
			startBuild := time.Now()
			if fd.buildBatch != nil {
				qb, sz, err := fd.buildBatch(sweep, sample)
				return nil, qb, sz, time.Since(startBuild), err
			}
			ie, sz, err := fd.build(sweep, sample)
			return ie, nil, sz, time.Since(startBuild), err
		}
		buildOnce := func(sample [][2]uint64) {
			if built {
				return
			}
			built = true
			if !warmedUp {
				warmKeys := keys[:1<<10]
				if wIE, wQB, _, _, werr := invokeBuild(sample); werr == nil {
					if wIE != nil {
						_ = wIE(warmKeys[0], warmKeys[len(warmKeys)-1])
					} else if wQB != nil {
						_ = wQB([][2]uint64{{warmKeys[0], warmKeys[len(warmKeys)-1]}})
					}
				}
				runtime.GC()
				warmedUp = true
			}
			builtIsEmpty, builtQueryBatch, builtSizeBits, builtBuildDur, builtBuildErr = invokeBuild(sample)
		}

		// Per-K BPK budget exceeded → break sweep entirely (BPK is L-
		// independent, so once filter family exceeds 25 bits/key it's
		// out of the competitive range for any L).
		bpkExceeded := false

		for _, L := range rangeLens {
			if fd.skipLs[L] {
				continue
			}
			// Skip L values that already reached FPR=0 at a smaller
			// K — measuring them at higher K would just duplicate.
			// Only applies to K-sweep filters: their FPR is monotone in K
			// and a larger K just produces a strictly larger filter. For
			// bpk-sweep filters (Grafite/SNARF/Rosetta) we still want a
			// row at every bpk target — Grafite's lossless fallback in
			// particular produces the same EF regardless of the requested
			// bpk, but headline plots need a row at sweep=10 to render.
			if fd.sweepName == "K" && lSaturated[L] {
				continue
			}

			params := b6Params{
				NKeys:         n,
				Eps:           eps,
				RangeLen:      L,
				QueryCount:    queryCount,
				QueryStrategy: mix.queryStrategy,
				QuerySeed:     int64(L) + 7777777,
				Distribution:  dist,
				Filter:        fd.name,
				SweepName:     fd.sweepName,
				SweepParam:    sweep,
				Parallelism:   parallelism,
			}
			paramsHash := params.hash()

			if !force {
				if cached := store.cachedRow(dist, fd.name, L, fd.sweepName, sweep, paramsHash); cached != nil {
					rows = append(rows, *cached)
					b6Logf("%-11s | %-14s | L=%-5d | %s=%-9.4g | %-9s | %-13s | %-13s | %-7s | %-9s  (cached)\n",
						dist, fd.name, L, fd.sweepName, sweep, "—", "—", "—", "—", "—")
					if cached.FPR == 0 {
						lSaturated[L] = true
					}
					if cached.BPKUsed > 25 {
						bpkExceeded = true
					}
					continue
				}
			}

			// Per-L Rosetta sample. Seed XOR keeps it independent from
			// the measurement-query seed so the sample is not the same
			// set of queries we then evaluate FPR against.
			var sampleQueries [][2]uint64
			if fd.lDependent {
				sampleSeed := params.QuerySeed ^ 0x055e77a
				sRng := rand.New(rand.NewSource(sampleSeed))
				sampleQueries = generateSmartQueriesWeighted(keys, 4096, L, mix.weights, sRng)
			}

			var (
				isEmpty    func(a, b uint64) bool
				queryBatch func([][2]uint64) []bool
				sizeBits   uint64
				buildErr   error
				buildDur   time.Duration
			)
			if fd.lDependent {
				// Rebuild per L with the L-specific sample.
				isEmpty, queryBatch, sizeBits, buildDur, buildErr = invokeBuild(sampleQueries)
			} else {
				buildOnce(nil)
				isEmpty = builtIsEmpty
				queryBatch = builtQueryBatch
				sizeBits = builtSizeBits
				buildErr = builtBuildErr
				buildDur = builtBuildDur
			}

			if buildErr != nil {
				rows = append(rows, b6Row{
					Distribution: dist,
					Filter:       fd.name,
					RangeLen:     L,
					SweepName:    fd.sweepName,
					SweepParam:   sweep,
					ParamsHash:   paramsHash,
					Note:         buildErr.Error(),
				})
				b6Logf("%-11s | %-14s | L=%-5d | %s=%-9.4g | %-9s | %-13s | %-13s | %-7s | %-9s  %s\n",
					dist, fd.name, L, fd.sweepName, sweep, "—", "—", "—", "—", "—", buildErr.Error())
				continue
			}
			buildMKeys := float64(n) / buildDur.Seconds() / 1e6
			actualBPK := float64(sizeBits) / float64(n)

			qrng := rand.New(rand.NewSource(params.QuerySeed))
			batch := generateSmartQueriesWeighted(keys, queryCount, L, mix.weights, qrng)
			if len(batch) == 0 {
				t.Logf("%s/%s L=%d %s=%.4g: smart-query generator returned 0 queries; skipping",
					dist, fd.name, L, fd.sweepName, sweep)
				continue
			}
			var (
				falsePositives int
				qDur           time.Duration
			)
			if queryBatch != nil {
				falsePositives, qDur = runQueriesBatchParallel(batch, queryBatch, parallelism)
			} else {
				falsePositives, qDur = runQueriesParallel(batch, isEmpty, parallelism)
			}
			nsPerQuery := float64(qDur.Nanoseconds()) / float64(len(batch))
			fpr := float64(falsePositives) / float64(len(batch))

			rows = append(rows, b6Row{
				Distribution:  dist,
				Filter:        fd.name,
				RangeLen:      L,
				BuildNs:       buildDur.Nanoseconds(),
				BuildMKeysSec: buildMKeys,
				QueryNsPerOp:  nsPerQuery,
				BPKUsed:       actualBPK,
				SizeBits:      sizeBits,
				FPR:           fpr,
				QueriesEmpty:  len(batch),
				SweepName:     fd.sweepName,
				SweepParam:    sweep,
				Parallelism:   parallelism,
				ParamsHash:    paramsHash,
			})
			b6Logf("%-11s | %-14s | L=%-5d | %s=%-9.4g | P=%-2d | %-9.1f | %-13.2f | %-13.1f | %-7.2f | %-9.4f\n",
				dist, fd.name, L, fd.sweepName, sweep, parallelism,
				float64(buildDur.Milliseconds()), buildMKeys, nsPerQuery, actualBPK, fpr)

			if fpr == 0 {
				lSaturated[L] = true
			}
			if actualBPK > 25 {
				bpkExceeded = true
			}
		}

		// Release the shared filter promptly before the next sweep
		// builds — otherwise GC may delay reclaim until the function
		// returns, stacking memory across sweeps.
		builtIsEmpty = nil
		builtQueryBatch = nil
		built = false
		runtime.GC()

		if fd.sweepName == "K" {
			// Two break conditions, both L-aggregate:
			//  - bpkExceeded: filter is past its memory budget for any
			//    L. Higher K won't help.
			//  - all L saturated: every L hit FPR=0; nothing to learn
			//    from larger K.
			if bpkExceeded {
				break
			}
			allDone := true
			for _, L := range rangeLens {
				if fd.skipLs[L] {
					continue
				}
				if !lSaturated[L] {
					allDone = false
					break
				}
			}
			if allDone {
				break
			}
		}
	}
	return rows
}

// isCGoFilter returns true for filter names backed by CGo wrappers
// (Grafite, SNARF, SuRF*). Their underlying C++ state is shared across
// IsEmpty calls and not concurrent-safe; we skip them at P>1.
func isCGoFilter(name string) bool {
	switch name {
	case "Grafite", "SNARF", "SuRFNone", "SuRFHash", "SuRFReal", "Rosetta":
		return true
	}
	return false
}

// parseB6Parallelism reads B6_PARALLEL env var. Default 1 = serial query
// loop (back-compatible with the existing JSON cells). Higher values split
// the query batch among that many goroutines; the filter is read-only
// after build so this is thread-safe. Used to study cache contention by
// running the same cell at P=1, 4, 16 and comparing wall-clock ns/op.
func parseB6Parallelism() int {
	v := os.Getenv("B6_PARALLEL")
	if v == "" {
		return 1
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 1 {
		return 1
	}
	return n
}

// runQueriesParallel splits batch into P contiguous chunks and runs them
// concurrently. Returns total false-positive count and wall-clock
// duration of the slowest chunk (≈ effective query time when P > 1).
func runQueriesParallel(
	batch [][2]uint64,
	isEmpty func(a, b uint64) bool,
	parallelism int,
) (int, time.Duration) {
	if parallelism <= 1 {
		fp := 0
		start := time.Now()
		for _, q := range batch {
			if !isEmpty(q[0], q[1]) {
				fp++
			}
		}
		return fp, time.Since(start)
	}

	chunk := (len(batch) + parallelism - 1) / parallelism
	fpCounts := make([]int, parallelism)
	var wg sync.WaitGroup
	start := time.Now()
	for w := 0; w < parallelism; w++ {
		lo := w * chunk
		if lo >= len(batch) {
			break
		}
		hi := lo + chunk
		if hi > len(batch) {
			hi = len(batch)
		}
		wg.Add(1)
		go func(idx int, qs [][2]uint64) {
			defer wg.Done()
			c := 0
			for _, q := range qs {
				if !isEmpty(q[0], q[1]) {
					c++
				}
			}
			fpCounts[idx] = c
		}(w, batch[lo:hi])
	}
	wg.Wait()
	dur := time.Since(start)
	total := 0
	for _, c := range fpCounts {
		total += c
	}
	return total, dur
}

// runQueriesBatchParallel is the buildBatch counterpart of runQueriesParallel.
// It runs `queryBatch` over the full batch (or per sub-batch under P>1) so a
// CGo filter pays one CGo crossing per chunk instead of per query — the only
// way to measure CGo per-query latency representatively. Semantics match the
// per-query path: a `false` result counts as a false positive (the smart-mix
// generator guarantees every query is empty).
func runQueriesBatchParallel(
	batch [][2]uint64,
	queryBatch func([][2]uint64) []bool,
	parallelism int,
) (int, time.Duration) {
	if parallelism <= 1 {
		start := time.Now()
		results := queryBatch(batch)
		dur := time.Since(start)
		fp := 0
		for _, r := range results {
			if !r {
				fp++
			}
		}
		return fp, dur
	}

	chunk := (len(batch) + parallelism - 1) / parallelism
	fpCounts := make([]int, parallelism)
	var wg sync.WaitGroup
	start := time.Now()
	for w := 0; w < parallelism; w++ {
		lo := w * chunk
		if lo >= len(batch) {
			break
		}
		hi := lo + chunk
		if hi > len(batch) {
			hi = len(batch)
		}
		wg.Add(1)
		go func(idx int, qs [][2]uint64) {
			defer wg.Done()
			res := queryBatch(qs)
			c := 0
			for _, r := range res {
				if !r {
					c++
				}
			}
			fpCounts[idx] = c
		}(w, batch[lo:hi])
	}
	wg.Wait()
	dur := time.Since(start)
	total := 0
	for _, c := range fpCounts {
		total += c
	}
	return total, dur
}
