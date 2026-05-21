//go:build heavy

// pointlat_runner_test.go — single-point latency at a source-mix operating
// point, measured under any other workload (typically smart_mix_mixed).
//
// Why this exists. TestB6IndustryLatency does a full FPR-vs-BPK sweep under
// the active B6_QUERY_MIX. TestB6BisectOperatingPoint bisects FPR under the
// active B6_QUERY_MIX to find an operating point in that workload. Neither
// reuses an existing operating point from a different mix — they each pick
// their own.
//
// For the right half of Tables 6.8/9/10 we want the opposite: take the
// operating point already established under guaranteed-empty queries
// (smart_mix), and measure ns/op of that exact same filter configuration
// under realistic queries (smart_mix_mixed). The "same operating point as
// Table~\ref{tab:bpk-L1}" claim in the caption then holds.
//
// Algorithm per (filter, dist, L):
//   1. Read smart_mix cache (b6_latency_N{n}_{source}/{filter}.json — bare
//      path when source = smart_mix).
//   2. Filter rows for (dist, L), sort by sweepParam, apply monotone
//      running-min on FPR.
//   3. Find the first row with FPR ≤ target — its sweepParam is the
//      operating point. (Matches the recipe used by the table-extraction
//      script in bench_results/extract_operating_points.py.)
//   4. If no such row exists in the source cache, skip — the cell is "---"
//      in the source already and stays "---" on the right.
//   5. Otherwise build the filter at that sweepParam, run the active query
//      mix (smart_mix_mixed) with the usual time budget, write one row to
//      b6_pointlat_N{n}_{mix}/{filter}.json.
//
// Cache reuse: this test never re-measures a cell whose row already exists
// in the destination cache with a matching params hash, so reruns are cheap
// (each cell ≤ one extra build).
//
// Env vars:
//   - B6_SOURCE_MIX  source cache to read operating points from
//                    (default: "smart_mix")
//   - B6_QUERY_MIX   workload to measure latency under
//                    (required, must be ≠ source; typical:
//                    smart_mix_mixed or gap_heavy_mixed)
//   - B6_BISECT_EPS  target FPR for operating-point detection
//                    (default: 0.001)
//   - reused: B6_N, B6_FILTERS_ONLY, B6_DISTS_ONLY, B6_RANGE_LENS,
//     SKIP_FILTERS, B6_PARALLELISM, B6_MEM.

package bench_test

import (
	"encoding/json"
	"fmt"
	mathbits "math/bits"
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

// pointLatStore caches single-point measurements at
// b6_pointlat_N{n}_{mix}/<filter>.json. Same on-disk shape as b6FilterDoc;
// reusing it lets extract_operating_points.py read this cache too (with
// --prefix b6_pointlat).
type pointLatStore struct {
	dir   string
	docs  map[string]*b6FilterDoc
	dirty map[string]bool
}

func newPointLatStore(nKeys int, mixSuffix string) *pointLatStore {
	dir := fmt.Sprintf("../bench_results/data/b6_pointlat_N%d", nKeys)
	if mixSuffix != "" {
		dir = fmt.Sprintf("../bench_results/data/b6_pointlat_N%d_%s", nKeys, mixSuffix)
	}
	return &pointLatStore{
		dir:   dir,
		docs:  make(map[string]*b6FilterDoc),
		dirty: make(map[string]bool),
	}
}

func (s *pointLatStore) loadFilter(filter string) *b6FilterDoc {
	if d, ok := s.docs[filter]; ok {
		return d
	}
	doc := &b6FilterDoc{
		Type:          "b6_pointlat_filter",
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

func (s *pointLatStore) cachedRow(dist, filter string, L uint64, sweepName string, sweep float64, paramsHash string) *b6Row {
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

func (s *pointLatStore) update(filter string, row b6Row) {
	doc := s.loadFilter(filter)
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

func (s *pointLatStore) flush() error {
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

// sourceCacheDir returns the directory holding the source-mix cache.
// smart_mix lives at bare b6_latency_N{n}/, every other mix is suffixed.
func sourceCacheDir(nKeys int, sourceMix string) string {
	if sourceMix == "" || sourceMix == "smart_mix" {
		return fmt.Sprintf("../bench_results/data/b6_latency_N%d", nKeys)
	}
	return fmt.Sprintf("../bench_results/data/b6_latency_N%d_%s", nKeys, sourceMix)
}

// readSourceFilter loads <filter>.json from the source-mix cache directory.
// Returns nil on any I/O or parse error; the caller skips the cell.
func readSourceFilter(nKeys int, sourceMix, filter string) *b6FilterDoc {
	path := filepath.Join(sourceCacheDir(nKeys, sourceMix), filter+".json")
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var doc b6FilterDoc
	if err := json.Unmarshal(buf, &doc); err != nil {
		return nil
	}
	return &doc
}

// operatingPointSweep mirrors operating_point() in extract_operating_points.py:
// monotone-min on FPR, then the smallest sweepParam with FPR ≤ targetFPR.
// We return that sweepParam (not the interpolated BPK) because we need to
// build the filter at a real grid value, not at a fractional bpk that the
// constructor cannot accept.
func operatingPointSweep(doc *b6FilterDoc, dist string, L uint64, targetFPR float64) (sweep float64, ok bool) {
	var rows []b6Row
	for _, r := range doc.Rows {
		if r.Distribution == dist && r.RangeLen == L && r.Note == "" {
			rows = append(rows, r)
		}
	}
	if len(rows) == 0 {
		return 0, false
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].SweepParam < rows[j].SweepParam })
	// Monotone running-min on FPR.
	var mono []b6Row
	minFPR := 1.0
	for _, r := range rows {
		if r.FPR <= minFPR {
			mono = append(mono, r)
			minFPR = r.FPR
		}
	}
	for _, r := range mono {
		if r.FPR <= targetFPR {
			return r.SweepParam, true
		}
	}
	return 0, false
}

// TestB6PointLatencyAtOperatingPoint reads operating points from
// B6_SOURCE_MIX and measures one latency point per cell under B6_QUERY_MIX.
// See the file-level comment for the algorithm.
func TestB6PointLatencyAtOperatingPoint(t *testing.T) {
	t.Cleanup(closeB6ProgressLog)

	const queryCount = 1 << 18
	eps := 0.01 // run-level eps (cache key)

	targetFPR := 0.001
	if v := strings.TrimSpace(os.Getenv("B6_BISECT_EPS")); v != "" {
		if x, err := strconv.ParseFloat(v, 64); err == nil && x > 0 {
			targetFPR = x
		}
	}

	sourceMix := strings.TrimSpace(os.Getenv("B6_SOURCE_MIX"))
	if sourceMix == "" {
		sourceMix = "smart_mix"
	}

	nValues := parseB6N()
	mix := parseB6QueryMix()
	if mix.name == sourceMix || (mix.name == "" && sourceMix == "smart_mix") {
		t.Fatalf("set B6_QUERY_MIX to a mix different from B6_SOURCE_MIX=%s "+
			"(e.g. smart_mix_mixed); current B6_QUERY_MIX=%q", sourceMix, mix.name)
	}
	t.Logf("source=%s query=%s targetFPR=%g", sourceMix, mix.name, targetFPR)

	rangeLens := []uint64{1, 16, 128, 1024}
	if rl := parseB6RangeLens(); rl != nil {
		rangeLens = rl
		t.Logf("B6_RANGE_LENS override: rangeLens=%v", rangeLens)
	}

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
	parallelism := parseB6Parallelism()

	for _, n := range nValues {
		n := n
		t.Run(fmt.Sprintf("N=2^%d", mathbits.TrailingZeros(uint(n))), func(t *testing.T) {
			store := newPointLatStore(n, mix.name)
			type cellOutcome struct {
				Dist, Filter string
				L            uint64
				Skipped      bool
				Reason       string
				Sweep        float64
				BPK          float64
				LatNs        float64
				ThrMK        float64
				FPRSource    float64 // FPR seen at operating point in source cache
				FPRMixed     float64 // FPR seen at operating point in this measurement
				FromCache    bool
			}
			var outcomes []cellOutcome

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
							continue
						}
						if parallelism > 1 && isCGoFilter(fd.name) {
							continue
						}
						srcDoc := readSourceFilter(n, sourceMix, fd.name)
						if srcDoc == nil {
							t.Logf("%s/%s: no source cache for mix=%s — skip whole filter",
								ds.name, fd.name, sourceMix)
							continue
						}
						t.Run(fd.name, func(t *testing.T) {
							for _, L := range rangeLens {
								if fd.skipLs[L] {
									continue
								}
								sweep, ok := operatingPointSweep(srcDoc, ds.name, L, targetFPR)
								if !ok {
									t.Logf("  %s/%s L=%-5d unreachable in source — skip",
										ds.name, fd.name, L)
									outcomes = append(outcomes, cellOutcome{
										Dist: ds.name, Filter: fd.name, L: L,
										Skipped: true, Reason: "unreachable in source",
									})
									continue
								}
								// Look up FPR seen at this sweep in the source for
								// the side-by-side report. (Optional — purely diagnostic.)
								var fprSource float64 = -1
								for _, r := range srcDoc.Rows {
									if r.Distribution == ds.name && r.RangeLen == L &&
										r.SweepParam == sweep {
										fprSource = r.FPR
										break
									}
								}

								params := b6Params{
									NKeys:         effN,
									Eps:           eps,
									RangeLen:      L,
									QueryCount:    queryCount,
									QueryStrategy: mix.queryStrategy,
									QuerySeed:     int64(L) + 7777777,
									Distribution:  ds.name,
									Filter:        fd.name,
									SweepName:     fd.sweepName,
									SweepParam:    sweep,
									Parallelism:   parallelism,
								}
								paramsHash := params.hash()

								if cached := store.cachedRow(ds.name, fd.name, L,
									fd.sweepName, sweep, paramsHash); cached != nil {
									t.Logf("  %s/%s L=%-5d %s=%-6.2f bpk=%-6.2f lat=%6.0fns thr=%5.1fM (cached)",
										ds.name, fd.name, L, fd.sweepName, sweep,
										cached.BPKUsed, cached.QueryNsPerOp, cached.BuildMKeysSec)
									outcomes = append(outcomes, cellOutcome{
										Dist: ds.name, Filter: fd.name, L: L,
										Sweep:     sweep,
										BPK:       cached.BPKUsed,
										LatNs:     cached.QueryNsPerOp,
										ThrMK:     cached.BuildMKeysSec,
										FPRSource: fprSource,
										FPRMixed:  cached.FPR,
										FromCache: true,
									})
									continue
								}

								row := bisectMeasureOnce(t, ds.name, fd, keys, L, sweep,
									queryCount, effN, eps, mix, parallelism)
								store.update(fd.name, row)
								_ = store.flush()
								if row.Note != "" {
									t.Logf("  %s/%s L=%-5d %s=%-6.2f BUILD ERR: %s",
										ds.name, fd.name, L, fd.sweepName, sweep, row.Note)
								} else {
									t.Logf("  %s/%s L=%-5d %s=%-6.2f bpk=%-6.2f lat=%6.0fns thr=%5.1fM "+
										"(fpr_src=%g fpr_mix=%g)",
										ds.name, fd.name, L, fd.sweepName, sweep,
										row.BPKUsed, row.QueryNsPerOp, row.BuildMKeysSec,
										fprSource, row.FPR)
								}
								outcomes = append(outcomes, cellOutcome{
									Dist: ds.name, Filter: fd.name, L: L,
									Sweep:     sweep,
									BPK:       row.BPKUsed,
									LatNs:     row.QueryNsPerOp,
									ThrMK:     row.BuildMKeysSec,
									FPRSource: fprSource,
									FPRMixed:  row.FPR,
								})
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

			// Final summary, easy to grep / paste into the tables.
			byFilter := map[string][]cellOutcome{}
			for _, o := range outcomes {
				byFilter[o.Filter] = append(byFilter[o.Filter], o)
			}
			filters := make([]string, 0, len(byFilter))
			for f := range byFilter {
				filters = append(filters, f)
			}
			sort.Strings(filters)
			t.Logf("\n=== point-latency at %s operating point under %s ===",
				sourceMix, mix.name)
			for _, f := range filters {
				os := byFilter[f]
				sort.Slice(os, func(i, j int) bool {
					if os[i].L != os[j].L {
						return os[i].L < os[j].L
					}
					return os[i].Dist < os[j].Dist
				})
				for _, o := range os {
					if o.Skipped {
						t.Logf("  %-9s | %-18s | L=%-5d | --- (%s)",
							f, o.Dist, o.L, o.Reason)
					} else {
						note := ""
						if o.FromCache {
							note = " [cached]"
						}
						t.Logf("  %-9s | %-18s | L=%-5d | sweep=%-6.2f | bpk=%-6.2f | lat=%6.0fns | thr=%5.1fM | fpr_src=%-9.4g fpr_mix=%-9.4g%s",
							f, o.Dist, o.L, o.Sweep, o.BPK, o.LatNs, o.ThrMK,
							o.FPRSource, o.FPRMixed, note)
					}
				}
			}
		})
	}
}
