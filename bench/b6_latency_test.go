package bench_test

import (
	"Thesis-bench-industry/thirdparty/snarf"
	"Thesis-bench-industry/thirdparty/surf"
	"Thesis/emptiness/approx/are_bloom"
	"Thesis/emptiness/approx/are_greedy_scan"
	"Thesis/emptiness/approx/are_hybrid_scan"
	"Thesis/emptiness/approx/are_soda_hash"
	"Thesis/emptiness/approx/are_trunc"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	mathbits "math/bits"
	"math/rand"
	"os"
	"runtime"
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
	const (
		queryCount = 1 << 18
		eps        = 0.01
	)
	rangeLens := []uint64{1, 16, 128, 1024, 4096, 16384, 65536}

	nValues := parseB6N()

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
				return loadSOSDUint64(syntheticDataPath("uniform_16M_uint64"), 0)
			}
		}},
		{"spread", func(n int) func() ([]uint64, error) {
			return func() ([]uint64, error) {
				return loadSOSDUint64(syntheticDataPath("spread_16M_uint64"), 0)
			}
		}},
		{"clustered", func(n int) func() ([]uint64, error) {
			return func() ([]uint64, error) {
				return loadSOSDUint64(syntheticDataPath("clustered_16M_uint64"), 0)
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
			store := newB6Store(n, queryCount, eps)
			fmt.Printf("\n=== B6: Build + Query latency + actual BPK + FPR, n=%d, ε=%.3f ===\n",
				n, eps)
			fmt.Printf("%-11s | %-14s | %-7s | %-13s | %-9s | %-13s | %-13s | %-7s | %-9s\n",
				"Distribution", "Filter", "L", "sweep", "build_ms", "build_Mkeys/s", "query_ns/op", "bpk", "fpr")

			for _, ds := range distributions {
				ds := ds
				t.Run(ds.name, func(t *testing.T) {
					allKeys, err := ds.makeLoad(n)()
					if err != nil {
						t.Skipf("load %s: %v", ds.name, err)
					}
					if len(allKeys) < n {
						t.Skipf("not enough keys for %s: have %d, need %d", ds.name, len(allKeys), n)
					}
					keys := allKeys[:n]
					keyBits := uint32(max(1, mathbits.Len64(keys[len(keys)-1])))
					t.Logf("%s: %d keys, range [%d, %d], keyBits=%d",
						ds.name, len(keys), keys[0], keys[n-1], keyBits)

					filters := buildB6Filters(keys, keyBits)
					for _, fd := range filters {
						fd := fd
						if skipSet[fd.name] {
							t.Logf("%s/%s: skipped via SKIP_FILTERS", ds.name, fd.name)
							continue
						}
						t.Run(fd.name, func(t *testing.T) {
							rows := runB6Filter(t, store, ds.name, fd,
								keys, rangeLens, queryCount, n, eps)
							store.update(ds.name, fd.name, rows)
							if err := store.flush(); err != nil {
								t.Errorf("flush %s: %v", store.path(), err)
							}
						})
					}
				})
			}

			t.Logf("wrote %s", store.path())
		})
	}
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
	b6SweepK        = []float64{4, 6, 8, 10, 12, 14, 16, 18, 20, 22}
	b6SweepBPK      = []float64{4, 6, 8, 10, 12, 14, 16, 18}
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
	// build returns isEmpty closure plus the actual filter footprint (bits).
	// sizeBits is then divided by n by the runner to get actual BPK. The
	// sweep value replaces whatever default the filter would otherwise use
	// for sweepName (eps/bpk/real_bits).
	build func(L uint64, sweep float64) (isEmpty func(a, b uint64) bool, sizeBits uint64, err error)
}

func buildB6Filters(keys []uint64, keyBits uint32) []b6FilterDef {
	return []b6FilterDef{
		{"SODA", "eps", b6SweepEps,
			func(L uint64, sweep float64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_soda_hash.NewSodaARE(keys, L, sweep)
				if err != nil {
					return nil, 0, err
				}
				return f.IsEmpty, f.SizeInBits(), nil
			}},
		{"Truncation", "K", b6SweepK,
			func(L uint64, sweep float64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_trunc.NewTruncARE(keys, keyBits, are_trunc.Config{K: uint32(sweep)})
				if err != nil {
					return nil, 0, err
				}
				return f.IsEmpty, f.SizeInBits(), nil
			}},
		{"Scan-ARE", "K", b6SweepK,
			func(L uint64, sweep float64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_hybrid_scan.NewHybridScanARE(keys, keyBits,
					are_hybrid_scan.Config{K: uint32(sweep)})
				if err != nil {
					return nil, 0, err
				}
				return f.IsEmpty, f.SizeInBits(), nil
			}},
		{"Greedy+Merge", "K", b6SweepK,
			func(L uint64, sweep float64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_greedy_scan.NewGreedyScanARE(keys, keyBits,
					are_greedy_scan.Config{K: uint32(sweep)})
				if err != nil {
					return nil, 0, err
				}
				return f.IsEmpty, f.SizeInBits(), nil
			}},
		{"BloomARE", "eps", b6SweepEps,
			func(L uint64, sweep float64) (func(a, b uint64) bool, uint64, error) {
				f, err := are_bloom.NewBloomARE(keys, L, sweep)
				if err != nil {
					return nil, 0, err
				}
				return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), nil
			}},
		{"Grafite", "bpk", b6SweepBPK,
			func(L uint64, sweep float64) (func(a, b uint64) bool, uint64, error) {
				f := tryGrafite(keys, sweep)
				if f == nil {
					return nil, 0, fmt.Errorf("grafite: target bpk=%.2f exceeds envelope", sweep)
				}
				return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), nil
			}},
		{"SNARF", "bpk", b6SweepBPK,
			func(L uint64, sweep float64) (func(a, b uint64) bool, uint64, error) {
				f := snarf.New(keys, sweep)
				return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), nil
			}},
		// SuRF is one filter family with three structural variants. We sweep
		// each variant's bit budget so the FPR-vs-BPK plots get a SuRF point
		// cloud across (suffixType, bitCount); the plotter folds all three
		// names into a single marker-only "SuRF" series.
		{"SuRFNone", "real_bits", b6SweepNoneBits,
			func(L uint64, sweep float64) (func(a, b uint64) bool, uint64, error) {
				f := surf.New(keys, surf.SuffixNone, 0, 0)
				return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), nil
			}},
		{"SuRFHash", "hash_bits", b6SweepHashBits,
			func(L uint64, sweep float64) (func(a, b uint64) bool, uint64, error) {
				f := surf.New(keys, surf.SuffixHash, int(sweep), 0)
				return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), nil
			}},
		{"SuRFReal", "real_bits", b6SweepRealBits,
			func(L uint64, sweep float64) (func(a, b uint64) bool, uint64, error) {
				f := surf.New(keys, surf.SuffixReal, 0, int(sweep))
				return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), nil
			}},
	}
}

type b6Row struct {
	Distribution  string  `json:"distribution"`
	Filter        string  `json:"filter"`
	RangeLen      uint64  `json:"rangeLen"`
	BuildNs       int64   `json:"buildNs"`
	BuildMKeysSec float64 `json:"buildMKeysSec"`
	QueryNsPerOp  float64 `json:"queryNsPerOp"` // wall-clock ns / total queries (with parallelism)
	BPKUsed       float64 `json:"bpkUsed"`
	SizeBits      uint64  `json:"sizeBits"`
	FPR           float64 `json:"fpr"`
	QueriesEmpty  int     `json:"queriesEmpty"`
	SweepName     string  `json:"sweepName"`
	SweepParam    float64 `json:"sweepParam"`
	Parallelism   int     `json:"parallelism"`
	ParamsHash    string  `json:"paramsHash"`
	Note          string  `json:"note,omitempty"`
}

// b6Params captures the hyperparameters that, if changed, invalidate a
// cached row. Stored hashed into b6Row.ParamsHash; the runner computes the
// hash for the current run and skips rows whose stored hash matches.
type b6Params struct {
	NKeys         int     `json:"nKeys"`
	Eps           float64 `json:"eps"`
	RangeLen      uint64  `json:"rangeLen"`
	QueryCount    int     `json:"queryCount"`
	QueryStrategy string  `json:"queryStrategy"`
	QuerySeed     int64   `json:"querySeed"`
	Distribution  string  `json:"distribution"`
	Filter        string  `json:"filter"`
	SweepName     string  `json:"sweepName"`
	SweepParam    float64 `json:"sweepParam"`
	Parallelism   int     `json:"parallelism"`
}

func (p b6Params) hash() string {
	buf, _ := json.Marshal(p)
	sum := sha256.Sum256(buf)
	return hex.EncodeToString(sum[:8]) // 16 hex chars is plenty for collisions
}

type b6Doc struct {
	Type          string  `json:"type"`
	NKeys         int     `json:"nKeys"`
	QueryCount    int     `json:"queryCount"`
	QueryStrategy string  `json:"queryStrategy"`
	Eps           float64 `json:"eps"`
	Timestamp     string  `json:"timestamp"`
	GitCommit     string  `json:"gitCommit"`
	Rows          []b6Row `json:"rows"`
}

// b6Store is the per-(distribution,filter) incremental writer. On first use
// it loads any prior b6_latency.json so re-running a single subtest merges
// into the prior run's output. Legacy rows lacking SweepName are dropped on
// load — they predate the K-sweep schema and would clutter plots.
type b6Store struct {
	mu  sync.Mutex
	doc b6Doc
}

func newB6Store(nKeys, queryCount int, eps float64) *b6Store {
	s := &b6Store{
		doc: b6Doc{
			Type:          "b6_latency",
			NKeys:         nKeys,
			QueryCount:    queryCount,
			QueryStrategy: "smart_mix_guaranteed_empty",
			Eps:           eps,
		},
	}
	if buf, err := os.ReadFile(s.path()); err == nil {
		var prior b6Doc
		if err := json.Unmarshal(buf, &prior); err == nil {
			for _, r := range prior.Rows {
				if r.SweepName == "" {
					continue
				}
				s.doc.Rows = append(s.doc.Rows, r)
			}
		}
	}
	return s
}

func (s *b6Store) path() string {
	return fmt.Sprintf("../bench_results/data/b6_latency_N%d.json", s.doc.NKeys)
}

func (s *b6Store) update(dist, filter string, rows []b6Row) {
	s.mu.Lock()
	defer s.mu.Unlock()
	// Only remove old rows whose paramsHash matches one of the incoming
	// rows. Rows for the same (dist, filter) but a different parameter
	// space (different parallelism, different sweep grid, etc.) are
	// preserved so a run at B6_PARALLEL=4 does not clobber the P=1 data.
	incoming := make(map[string]struct{}, len(rows))
	for _, r := range rows {
		incoming[r.ParamsHash] = struct{}{}
	}
	kept := s.doc.Rows[:0]
	for _, r := range s.doc.Rows {
		if r.Distribution == dist && r.Filter == filter {
			if _, replaced := incoming[r.ParamsHash]; replaced {
				continue
			}
		}
		kept = append(kept, r)
	}
	s.doc.Rows = append(kept, rows...)
}

// cachedRow returns a prior row for (dist, filter, L, sweepName, sweepParam)
// whose paramsHash matches the requested one, or nil. Use to short-circuit
// measurement when FORCE is unset.
func (s *b6Store) cachedRow(dist, filter string, L uint64, sweepName string, sweepParam float64, paramsHash string) *b6Row {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.doc.Rows {
		r := &s.doc.Rows[i]
		if r.Distribution == dist && r.Filter == filter &&
			r.RangeLen == L && r.SweepName == sweepName &&
			r.SweepParam == sweepParam && r.ParamsHash == paramsHash {
			return r
		}
	}
	return nil
}

func (s *b6Store) flush() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.doc.Timestamp = time.Now().UTC().Format(time.RFC3339)
	s.doc.GitCommit = gitCommitShort()
	if err := os.MkdirAll("../bench_results/data", 0755); err != nil {
		return err
	}
	buf, err := json.MarshalIndent(s.doc, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.path(), buf, 0644)
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
) []b6Row {
	force := os.Getenv("FORCE") != ""
	parallelism := parseB6Parallelism()

	rows := make([]b6Row, 0, len(rangeLens)*len(fd.sweepValues))
	warmedUp := false
	for _, L := range rangeLens {
		for _, sweep := range fd.sweepValues {
			params := b6Params{
				NKeys:         n,
				Eps:           eps,
				RangeLen:      L,
				QueryCount:    queryCount,
				QueryStrategy: "smart_mix_guaranteed_empty",
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
					fmt.Printf("%-11s | %-14s | L=%-5d | %s=%-9.4g | %-9s | %-13s | %-13s | %-7s | %-9s  (cached)\n",
						dist, fd.name, L, fd.sweepName, sweep, "—", "—", "—", "—", "—")
					continue
				}
			}

			// Lazy warm-up — only flush CGo init / page faults once we know we'll
			// actually measure something this iteration.
			if !warmedUp {
				warmKeys := keys[:1<<10]
				if warmIsEmpty, _, werr := fd.build(rangeLens[0], sweep); werr == nil {
					_ = warmIsEmpty(warmKeys[0], warmKeys[len(warmKeys)-1])
				}
				runtime.GC()
				warmedUp = true
			}

			startBuild := time.Now()
			isEmpty, sizeBits, err := fd.build(L, sweep)
			buildDur := time.Since(startBuild)
			if err != nil {
				rows = append(rows, b6Row{
					Distribution: dist,
					Filter:       fd.name,
					RangeLen:     L,
					SweepName:    fd.sweepName,
					SweepParam:   sweep,
					ParamsHash:   paramsHash,
					Note:         err.Error(),
				})
				fmt.Printf("%-11s | %-14s | L=%-5d | %s=%-9.4g | %-9s | %-13s | %-13s | %-7s | %-9s  %s\n",
					dist, fd.name, L, fd.sweepName, sweep, "—", "—", "—", "—", "—", err.Error())
				continue
			}
			buildMKeys := float64(n) / buildDur.Seconds() / 1e6
			actualBPK := float64(sizeBits) / float64(n)

			qrng := rand.New(rand.NewSource(params.QuerySeed))
			batch := generateSmartQueries(keys, queryCount, L, qrng)
			if len(batch) == 0 {
				t.Logf("%s/%s L=%d %s=%.4g: smart-query generator returned 0 queries; skipping",
					dist, fd.name, L, fd.sweepName, sweep)
				continue
			}

			// Smart queries are guaranteed empty → any false return is a
			// false positive. With parallelism > 1 we split the batch into
			// `parallelism` chunks and run them in goroutines; the filter
			// is read-only after build so this is safe. ns/op is computed
			// against wall-clock so it shows speedup (or cache contention)
			// at higher parallelism.
			falsePositives, qDur := runQueriesParallel(batch, isEmpty, parallelism)
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
			fmt.Printf("%-11s | %-14s | L=%-5d | %s=%-9.4g | P=%-2d | %-9.1f | %-13.2f | %-13.1f | %-7.2f | %-9.4f\n",
				dist, fd.name, L, fd.sweepName, sweep, parallelism,
				float64(buildDur.Milliseconds()), buildMKeys, nsPerQuery, actualBPK, fpr)
		}
	}
	return rows
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
