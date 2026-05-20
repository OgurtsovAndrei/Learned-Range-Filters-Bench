// bisect_runner_test.go — fast operating-point finder via binary search.
//
// TestB6IndustryLatency above does a full linear sweep over each filter's
// tuning parameter to draw FPR-vs-BPK curves. When all we need is the
// operating point (smallest sweep value with FPR <= target), the linear
// sweep wastes ~70 % of measurements. This runner does ~log2(grid) builds
// per (dist, filter, L) instead.
//
// Bracketing: start with [first, last] from fd.sweepValues; measure both.
//   - If both above eps  → not reachable, report ---
//   - If both below eps  → return the smallest sweep value (op pt is <= it,
//                          but the sweep doesn't go lower)
//   - Otherwise          → binary-search inside the bracket
//
// Final operating point is the log-linear interpolation between the last
// straddling pair, matching extract_operating_points.py.
//
// Cache: bench_results/data/b6_bisect_N{n}_{mix}/<filter>.json, same row
// shape as the sweep cache so extract_operating_points.py works as-is.
// Cache is keyed by full b6Params hash including SweepParam, so the same
// midpoint never re-measures across runs.
//
// Env vars:
//   - B6_BISECT_EPS      target FPR (default 0.001)
//   - B6_BISECT_ITERS    max bisect iterations (default 8)
//   - B6_BISECT_TOL      stop when |hi - lo| <= tol (default 0.5 for float
//                        sweeps, 1 for integer sweeps)
//   - Reuses: B6_N, B6_QUERY_MIX, B6_FILTERS_ONLY, B6_DISTS_ONLY,
//     B6_RANGE_LENS, SKIP_FILTERS.

package bench_test

import (
	"Thesis/utils"
	"encoding/json"
	"fmt"
	"math"
	mathbits "math/bits"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

// bisectConfig holds the tunable knobs for the binary search.
type bisectConfig struct {
	eps      float64
	maxIters int
	tol      float64
}

func parseBisectConfig() bisectConfig {
	cfg := bisectConfig{eps: 0.001, maxIters: 8, tol: 0.5}
	if v := strings.TrimSpace(os.Getenv("B6_BISECT_EPS")); v != "" {
		if x, err := strconv.ParseFloat(v, 64); err == nil && x > 0 {
			cfg.eps = x
		}
	}
	if v := strings.TrimSpace(os.Getenv("B6_BISECT_ITERS")); v != "" {
		if x, err := strconv.Atoi(v); err == nil && x > 0 {
			cfg.maxIters = x
		}
	}
	if v := strings.TrimSpace(os.Getenv("B6_BISECT_TOL")); v != "" {
		if x, err := strconv.ParseFloat(v, 64); err == nil && x > 0 {
			cfg.tol = x
		}
	}
	return cfg
}

// sweepIsInteger reports whether a sweep parameter only takes integer
// values (K for SODA-family, *_bits for SuRF variants).
func sweepIsInteger(sweepName string) bool {
	return sweepName == "K" || strings.HasSuffix(sweepName, "_bits")
}

// roundSweep snaps mid to the parameter's natural grid (integer for K /
// *_bits, half-integer for bpk to give finer resolution).
func roundSweep(sweepName string, mid float64) float64 {
	if sweepIsInteger(sweepName) {
		return float64(int64(mid + 0.5))
	}
	// BPK: snap to half-integers — enough resolution for the table without
	// drifting too far from the predefined grid (which is on integer BPK).
	return float64(int64(mid*2+0.5)) / 2
}

// bisectStore is a thin per-filter cache, mirroring b6Store's on-disk
// shape but rooted at b6_bisect_N{n}_{mix}/. We don't reuse b6Store
// because its path() hardcodes the b6_latency prefix.
type bisectStore struct {
	dir   string
	docs  map[string]*b6FilterDoc
	dirty map[string]bool
}

func newBisectStore(nKeys, queryCount int, eps float64, queryStrategy, mixSuffix string) *bisectStore {
	dir := fmt.Sprintf("../bench_results/data/b6_bisect_N%d", nKeys)
	if mixSuffix != "" {
		dir = fmt.Sprintf("../bench_results/data/b6_bisect_N%d_%s", nKeys, mixSuffix)
	}
	return &bisectStore{
		dir:   dir,
		docs:  make(map[string]*b6FilterDoc),
		dirty: make(map[string]bool),
	}
}

func (s *bisectStore) loadFilter(filter string) *b6FilterDoc {
	if d, ok := s.docs[filter]; ok {
		return d
	}
	doc := &b6FilterDoc{
		Type:          "b6_bisect_filter",
		Filter:        filter,
		SchemaVersion: b6SchemaVersion,
	}
	path := filepath.Join(s.dir, filter+".json")
	if buf, err := os.ReadFile(path); err == nil {
		var d b6FilterDoc
		if err := json.Unmarshal(buf, &d); err == nil {
			doc = &d
		}
	}
	s.docs[filter] = doc
	return doc
}

// cachedRow finds a row in the loaded doc matching the full params hash.
// Used to skip already-measured (dist, filter, L, sweep) combinations.
func (s *bisectStore) cachedRow(dist, filter string, L uint64, sweepName string, sweep float64, paramsHash string) *b6Row {
	doc := s.loadFilter(filter)
	for i := range doc.Rows {
		r := &doc.Rows[i]
		if r.Distribution == dist && r.RangeLen == L &&
			r.SweepName == sweepName && r.SweepParam == sweep &&
			r.ParamsHash == paramsHash {
			return r
		}
	}
	return nil
}

func (s *bisectStore) update(filter string, row b6Row) {
	doc := s.loadFilter(filter)
	// De-dup: if a row with the same params hash already exists, replace it
	// in place rather than appending; otherwise repeated reruns would grow
	// the file linearly.
	for i := range doc.Rows {
		if doc.Rows[i].ParamsHash == row.ParamsHash &&
			doc.Rows[i].Distribution == row.Distribution &&
			doc.Rows[i].RangeLen == row.RangeLen &&
			doc.Rows[i].SweepParam == row.SweepParam {
			doc.Rows[i] = row
			s.dirty[filter] = true
			return
		}
	}
	doc.Rows = append(doc.Rows, row)
	s.dirty[filter] = true
}

func (s *bisectStore) flush() error {
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return err
	}
	for f, dirty := range s.dirty {
		if !dirty {
			continue
		}
		doc := s.docs[f]
		doc.Timestamp = time.Now().UTC().Format(time.RFC3339)
		buf, err := json.MarshalIndent(doc, "", "  ")
		if err != nil {
			return err
		}
		path := filepath.Join(s.dir, f+".json")
		tmp := path + ".tmp"
		if err := os.WriteFile(tmp, buf, 0o644); err != nil {
			return err
		}
		if err := os.Rename(tmp, path); err != nil {
			return err
		}
		s.dirty[f] = false
	}
	return nil
}

// bisectMeasureOnce builds the filter at `sweep`, runs the FPR/latency
// query batch for `L`, and returns the populated b6Row. Mirrors the inner
// body of runB6Filter — no per-L sweep, no warm-up, no saturation tracking.
func bisectMeasureOnce(
	t *testing.T,
	dist string,
	fd b6FilterDef,
	keys []uint64,
	L uint64,
	sweep float64,
	queryCount int,
	nKeys int,
	eps float64,
	mix b6QueryMix,
	parallelism int,
) b6Row {
	params := b6Params{
		NKeys:         nKeys,
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
	row := b6Row{
		Distribution: dist,
		Filter:       fd.name,
		RangeLen:     L,
		SweepName:    fd.sweepName,
		SweepParam:   sweep,
		ParamsHash:   paramsHash,
		Parallelism:  parallelism,
	}

	// Build sample queries for L-dependent filters (Rosetta).
	var sampleQueries [][2]uint64
	if fd.lDependent {
		sampleSeed := params.QuerySeed ^ 0x055e77a
		sRng := rand.New(rand.NewSource(sampleSeed))
		sampleQueries = generateSmartQueriesWeighted(keys, 4096, L, mix.weights, sRng)
	}

	trackMem := os.Getenv("B6_MEM") != ""
	var monitor *utils.MemoryMonitor
	if trackMem {
		utils.ForceGC()
		monitor = utils.StartMemoryMonitor(time.Millisecond)
	}

	startBuild := time.Now()
	var (
		isEmpty    func(a, b uint64) bool
		queryBatch func([][2]uint64) []bool
		sizeBits   uint64
		err        error
	)
	if fd.buildBatch != nil {
		queryBatch, sizeBits, err = fd.buildBatch(sweep, sampleQueries)
	} else {
		isEmpty, sizeBits, err = fd.build(sweep, sampleQueries)
	}
	buildDur := time.Since(startBuild)

	var peakRSS uint64
	if trackMem {
		peakRSS = monitor.Stop()
	}

	if err != nil {
		row.Note = err.Error()
		return row
	}
	row.BuildNs = buildDur.Nanoseconds()
	row.BuildMKeysSec = float64(nKeys) / buildDur.Seconds() / 1e6
	row.BuildPeakRSSMB = float64(peakRSS) / (1024 * 1024)
	row.SizeBits = sizeBits
	row.BPKUsed = float64(sizeBits) / float64(nKeys)

	// Query batch — same logic as runB6Filter.
	qrng := rand.New(rand.NewSource(params.QuerySeed))
	var (
		batch      [][2]uint64
		emptyMask  []bool
		emptyCount int
	)
	if mix.allowNonEmpty {
		batch = generateMixedQueriesWeighted(keys, queryCount, L, mix.weights, qrng)
		emptyMask = make([]bool, len(batch))
		for i, q := range batch {
			idx := sort.Search(len(keys), func(k int) bool { return keys[k] >= q[0] })
			isE := idx >= len(keys) || keys[idx] > q[1]
			emptyMask[i] = isE
			if isE {
				emptyCount++
			}
		}
	} else {
		batch = generateSmartQueriesWeighted(keys, queryCount, L, mix.weights, qrng)
		emptyCount = len(batch)
	}
	if len(batch) == 0 {
		row.Note = "no queries"
		return row
	}

	var (
		falsePositives int
		totalQueried   int
		qDur           time.Duration
	)
	if queryBatch != nil {
		falsePositives, totalQueried, qDur = runQueriesBatchTimed(
			batch, emptyMask, queryBatch, parallelism, mix.timeBudget)
	} else {
		falsePositives, totalQueried, qDur = runQueriesTimed(
			batch, emptyMask, isEmpty, parallelism, mix.timeBudget)
	}
	row.QueryNsPerOp = float64(qDur.Nanoseconds()) / float64(totalQueried)

	emptyDenom := emptyCount
	if mix.timeBudget > 0 && emptyDenom > 0 {
		repeats := totalQueried / len(batch)
		if repeats < 1 {
			repeats = 1
		}
		emptyDenom = emptyCount * repeats
	}
	if emptyDenom > 0 {
		row.FPR = float64(falsePositives) / float64(emptyDenom)
	}
	row.QueriesEmpty = emptyCount
	return row
}

// bisectOutcome carries the per-(dist, filter, L) result.
type bisectOutcome struct {
	Dist      string
	Filter    string
	RangeLen  uint64
	Reachable bool
	BPK       float64 // interpolated operating-point BPK
	LatNs     float64 // latency at the first-below-eps measurement
	ThrMK     float64 // build throughput at the first-below-eps measurement
	NMeas     int     // total measurements consumed
	NCached   int     // measurements served from cache
	LowestFPR float64 // for unreachable cases — best FPR achieved
	LowestSwp float64 // sweep value at LowestFPR
}

// bisectOne runs binary search over fd.sweepValues for the given L. Each
// measurement is cached (de-duped against earlier identical params).
func bisectOne(
	t *testing.T,
	store *bisectStore,
	dist string,
	fd b6FilterDef,
	keys []uint64,
	L uint64,
	queryCount int,
	nKeys int,
	eps float64, // run-level eps, written into b6Params (NOT the bisect target)
	mix b6QueryMix,
	parallelism int,
	cfg bisectConfig,
) bisectOutcome {
	out := bisectOutcome{Dist: dist, Filter: fd.name, RangeLen: L, LowestFPR: 1.0}
	if len(fd.sweepValues) == 0 {
		return out
	}

	// Closure that measures (or fetches from cache) a single sweep value.
	measure := func(sweep float64) b6Row {
		params := b6Params{
			NKeys:         nKeys,
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
		if cached := store.cachedRow(dist, fd.name, L, fd.sweepName, sweep, paramsHash); cached != nil {
			out.NCached++
			t.Logf("  %s/%s L=%-5d %s=%-6.2f bpk=%-6.2f fpr=%-9.4g (cached)",
				dist, fd.name, L, fd.sweepName, sweep, cached.BPKUsed, cached.FPR)
			return *cached
		}
		out.NMeas++
		row := bisectMeasureOnce(t, dist, fd, keys, L, sweep, queryCount, nKeys, eps, mix, parallelism)
		store.update(fd.name, row)
		_ = store.flush() // best-effort; ignore intra-bisect flush errors
		t.Logf("  %s/%s L=%-5d %s=%-6.2f bpk=%-6.2f fpr=%-9.4g build=%.0fms",
			dist, fd.name, L, fd.sweepName, sweep, row.BPKUsed, row.FPR,
			float64(row.BuildNs)/1e6)
		return row
	}

	swValues := append([]float64(nil), fd.sweepValues...)
	sort.Float64s(swValues)
	lo := swValues[0]
	hi := swValues[len(swValues)-1]

	rLo := measure(lo)
	rHi := measure(hi)

	updateLowest := func(r b6Row) {
		if r.Note != "" {
			return
		}
		if r.FPR < out.LowestFPR {
			out.LowestFPR = r.FPR
			out.LowestSwp = r.BPKUsed
		}
	}
	updateLowest(rLo)
	updateLowest(rHi)

	// Both above eps → unreachable in this sweep range.
	if rHi.Note == "" && rHi.FPR > cfg.eps {
		return out
	}
	// Both below eps → operating point is at or below lo (we can't probe
	// lower than the sweep grid allows). Report lo as the operating point.
	if rLo.Note == "" && rLo.FPR <= cfg.eps {
		out.Reachable = true
		out.BPK = rLo.BPKUsed
		out.LatNs = rLo.QueryNsPerOp
		out.ThrMK = rLo.BuildMKeysSec
		return out
	}
	// rHi below eps, rLo above eps → bisect.
	bestBelow := rHi
	bestAbove := rLo
	isInt := sweepIsInteger(fd.sweepName)
	tol := cfg.tol
	if isInt && tol < 1 {
		tol = 1
	}

	for iter := 0; iter < cfg.maxIters; iter++ {
		if hi-lo <= tol {
			break
		}
		mid := roundSweep(fd.sweepName, (lo+hi)/2)
		if mid <= lo || mid >= hi {
			// Grid exhausted (integer sweeps).
			break
		}
		r := measure(mid)
		updateLowest(r)
		if r.Note != "" {
			// Build error — treat as "above eps" to push hi lower.
			hi = mid
			continue
		}
		if r.FPR <= cfg.eps {
			hi = mid
			bestBelow = r
		} else {
			lo = mid
			bestAbove = r
		}
	}

	out.Reachable = true
	// Final interpolation in log(FPR) space, identical to
	// extract_operating_points.py. If the above-pt measurement saturated
	// (FPR == 0) the interpolation is meaningless — report the measured
	// BPK directly.
	if bestAbove.Note != "" || bestAbove.FPR <= cfg.eps {
		out.BPK = bestBelow.BPKUsed
	} else if bestBelow.FPR <= 0 || bestBelow.FPR < cfg.eps*1e-3 {
		out.BPK = bestBelow.BPKUsed
	} else {
		b1, f1 := bestAbove.BPKUsed, bestAbove.FPR
		b2, f2 := bestBelow.BPKUsed, bestBelow.FPR
		if f1 == f2 {
			out.BPK = b2
		} else {
			out.BPK = b1 + (b2-b1)*
				(math.Log(f1)-math.Log(cfg.eps))/(math.Log(f1)-math.Log(f2))
		}
	}
	out.LatNs = bestBelow.QueryNsPerOp
	out.ThrMK = bestBelow.BuildMKeysSec
	return out
}

// TestB6BisectOperatingPoint runs the bisect operating-point finder for
// every (dist, filter, L) selected by the usual env vars. ~3× faster than
// the full sweep when only operating points are needed.
func TestB6BisectOperatingPoint(t *testing.T) {
	t.Cleanup(closeB6ProgressLog)

	const queryCount = 1 << 18
	eps := 0.01 // run-level eps (cache key); bisect target lives in cfg.eps.
	rangeLens := []uint64{1, 16, 128, 1024}

	cfg := parseBisectConfig()
	nValues := parseB6N()
	mix := parseB6QueryMix()
	if mix.name != "" {
		t.Logf("B6_QUERY_MIX=%s — cache routed to b6_bisect_N{n}_%s",
			mix.name, mix.name)
	}
	t.Logf("bisect cfg: eps=%g maxIters=%d tol=%g", cfg.eps, cfg.maxIters, cfg.tol)

	type distSpec struct {
		name     string
		makeLoad func(n int) func() ([]uint64, error)
	}
	distributions := []distSpec{
		{"sosd_fb", func(n int) func() ([]uint64, error) {
			return func() ([]uint64, error) { return loadFacebookKeys(2 * n) }
		}},
		{"sosd_wiki", func(n int) func() ([]uint64, error) {
			return func() ([]uint64, error) { return loadWikiKeys(2 * n) }
		}},
		{"sosd_osm", func(n int) func() ([]uint64, error) {
			return func() ([]uint64, error) { return loadOSMKeys(2 * n) }
		}},
		{"sosd_books", func(n int) func() ([]uint64, error) {
			return func() ([]uint64, error) { return loadBooksKeys(2 * n) }
		}},
		{"sosd_books_800m", func(n int) func() ([]uint64, error) {
			return func() ([]uint64, error) { return loadBooks800MKeys(2 * n) }
		}},
		{"uniform", func(n int) func() ([]uint64, error) {
			return func() ([]uint64, error) {
				return loadSOSDUint64(syntheticDataPath(syntheticFile("uniform", n)), 0)
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
	onlyFilters := parseCsvSet(os.Getenv("B6_FILTERS_ONLY"))
	onlyDists := parseCsvSet(os.Getenv("B6_DISTS_ONLY"))
	if rl := parseB6RangeLens(); rl != nil {
		rangeLens = rl
		t.Logf("B6_RANGE_LENS override: rangeLens=%v", rangeLens)
	}

	parallelism := parseB6Parallelism()

	for _, n := range nValues {
		n := n
		t.Run(fmt.Sprintf("N=2^%d", mathbits.TrailingZeros(uint(n))), func(t *testing.T) {
			store := newBisectStore(n, queryCount, eps, mix.queryStrategy, mix.name)
			var outcomes []bisectOutcome

			for _, ds := range distributions {
				ds := ds
				if onlyDists != nil && !onlyDists[ds.name] {
					continue
				}
				t.Run(ds.name, func(t *testing.T) {
					allKeys, err := ds.makeLoad(n)()
					if err != nil {
						t.Skipf("load %s: %v", ds.name, err)
					}
					effN := n
					if len(allKeys) < n {
						effN = len(allKeys)
					}
					if effN < 1<<10 {
						t.Skipf("dataset too small: %d", effN)
					}
					keys := allKeys[:effN]
					keyBits := uint32(max(1, mathbits.Len64(keys[len(keys)-1])))
					t.Logf("%s: %d keys, range [%d, %d], keyBits=%d",
						ds.name, len(keys), keys[0], keys[len(keys)-1], keyBits)

					filters := buildB6Filters(keys, keyBits)
					for _, fd := range filters {
						fd := fd
						if skipSet[fd.name] {
							continue
						}
						if onlyFilters != nil && !onlyFilters[fd.name] {
							continue
						}
						if fd.skipDists[ds.name] {
							t.Logf("%s/%s: skipped — known unsafe combination", ds.name, fd.name)
							continue
						}
						if parallelism > 1 && isCGoFilter(fd.name) {
							continue
						}
						t.Run(fd.name, func(t *testing.T) {
							for _, L := range rangeLens {
								if fd.skipLs[L] {
									continue
								}
								out := bisectOne(t, store, ds.name, fd, keys, L,
									queryCount, effN, eps, mix, parallelism, cfg)
								outcomes = append(outcomes, out)
								if out.Reachable {
									t.Logf("  → %s/%s L=%-5d BPK=%-6.2f lat=%.0fns thr=%.1fM  (meas=%d cached=%d)",
										ds.name, fd.name, L, out.BPK, out.LatNs, out.ThrMK,
										out.NMeas, out.NCached)
								} else {
									t.Logf("  → %s/%s L=%-5d --- (lowest FPR=%g @ bpk=%.2f, meas=%d cached=%d)",
										ds.name, fd.name, L, out.LowestFPR, out.LowestSwp,
										out.NMeas, out.NCached)
								}
							}
						})
						runtime.GC()
						debug.FreeOSMemory()
					}
				})
			}

			if err := store.flush(); err != nil {
				t.Errorf("flush: %v", err)
			}
			summariseBisectOutcomes(t, outcomes, cfg.eps)
		})
	}
}

// summariseBisectOutcomes prints a final operating-point table grouped by
// (filter, L). Mirrors the layout consumed by extract_operating_points.py
// so the result is easy to copy into evaluation_tables.tex.
func summariseBisectOutcomes(t *testing.T, outcomes []bisectOutcome, eps float64) {
	if len(outcomes) == 0 {
		return
	}
	// Group by filter then L for the printout.
	byFilter := map[string][]bisectOutcome{}
	for _, o := range outcomes {
		byFilter[o.Filter] = append(byFilter[o.Filter], o)
	}
	filters := make([]string, 0, len(byFilter))
	for f := range byFilter {
		filters = append(filters, f)
	}
	sort.Strings(filters)

	var totalMeas, totalCached int
	t.Logf("\n=== bisect operating points (target FPR ≤ %g) ===", eps)
	for _, f := range filters {
		os := byFilter[f]
		sort.Slice(os, func(i, j int) bool {
			if os[i].RangeLen != os[j].RangeLen {
				return os[i].RangeLen < os[j].RangeLen
			}
			return os[i].Dist < os[j].Dist
		})
		for _, o := range os {
			totalMeas += o.NMeas
			totalCached += o.NCached
			if o.Reachable {
				t.Logf("  %-9s | %-18s | L=%-5d | BPK=%-6.2f | lat=%6.0fns | thr=%5.1fM | meas=%d cached=%d",
					f, o.Dist, o.RangeLen, o.BPK, o.LatNs, o.ThrMK, o.NMeas, o.NCached)
			} else {
				t.Logf("  %-9s | %-18s | L=%-5d | ---     (lowest FPR=%g @ bpk=%.2f) | meas=%d cached=%d",
					f, o.Dist, o.RangeLen, o.LowestFPR, o.LowestSwp, o.NMeas, o.NCached)
			}
		}
	}
	t.Logf("\nbisect totals: %d new measurements, %d cache hits", totalMeas, totalCached)
}
