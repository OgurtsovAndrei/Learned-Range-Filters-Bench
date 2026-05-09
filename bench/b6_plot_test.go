package bench_test

import (
	"Thesis/testutils"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// b6SeriesStyles extends DefaultSeriesStyles with B6-only filters (Rosetta).
//
// Filters not listed here fall through to a default gray rendering.
var b6SeriesStyles = func() map[string]SeriesStyle {
	m := make(map[string]SeriesStyle, len(DefaultSeriesStyles)+2)
	for k, v := range DefaultSeriesStyles {
		m[k] = v
	}
	// Rosetta — dark green (Tailwind green-700), readable on white projector,
	// distinct from teal Grafite family. Diamond marker matches the industry-
	// baseline visual convention (Grafite, SNARF, SuRF*).
	m["Rosetta"] = SeriesStyle{Name: "Rosetta", Color: "#15803d", Marker: "diamond"}
	// SuRF is one family rendered as a marker-only point cloud across all
	// three structural variants (None / Hash / Real). Inherit the
	// SuRFReal(8) palette so plots stay consistent with comparison_test.go.
	surfColor := "#dc2626"
	surfMarker := "diamond"
	if s, ok := DefaultSeriesStyles["SuRFReal(8)"]; ok {
		surfColor = s.Color
		surfMarker = s.Marker
	}
	m["SuRF"] = SeriesStyle{Name: "SuRF", Color: surfColor, Marker: surfMarker}
	// SODA-PEF — same filter as SODA but with the PEF (Partitioned Elias-Fano)
	// backend. Amber square to sit visually next to SODA (yellow circle).
	m["SODA-PEF"] = SeriesStyle{Name: "SODA-PEF", Color: "#f59e0b", Marker: "square"}
	// Scan-ARE-SODA-PEF — Scan-ARE with SODA fallback and PEF backend.
	// Deep purple square, sibling to Scan-ARE-SODA (purple triangle).
	m["Scan-ARE-SODA-PEF"] = SeriesStyle{Name: "Scan-ARE-SODA-PEF", Color: "#7c3aed", Marker: "square"}
	// Scan-ARE-SODA-FbPEF — PEF only in fallback, clusters use OneD.
	// Lighter purple to sit between Scan-ARE-SODA and Scan-ARE-SODA-PEF.
	m["Scan-ARE-SODA-FbPEF"] = SeriesStyle{Name: "Scan-ARE-SODA-FbPEF", Color: "#a78bfa", Marker: "triangle"}
	// Keep SuRFReal entry as a fallback used only by the per-(metric, dist)
	// plots that pick a single representative cell; same palette as SuRF.
	m["SuRFReal"] = SeriesStyle{Name: "SuRFReal", Color: surfColor, Marker: surfMarker}
	return m
}()

// b6 series rendering order — industry baselines (cool palette) first, then
// the academic line (warm palette) ending with Bloom as the slope reference.
var b6PlotOrder = []string{
	"Grafite",
	"SNARF",
	"SuRF",
	"Rosetta",
	"SODA",
	"SODA-PEF",
	"Scan-ARE-Trunc",
	"Scan-ARE-SODA",
	"Scan-ARE-SODA-PEF",
	"Scan-ARE-SODA-FbPEF",
	"Greedy+Merge-Trunc",
	"Greedy+Merge-SODA",
	"BloomARE",
}

// Headline sweep values used for the existing per-(metric, distribution)
// plots and for the L-trajectory trade-off plot. We pick a single sweep
// value per filter family so each filter renders as one curve through L.
const (
	b6HeadlineEps     = 0.01
	b6HeadlineK       = 14.0 // Roughly equivalent to eps≈0.01 across our filters.
	b6HeadlineBPK     = 10.0
	b6HeadlineRealBit = 8.0
)

// matchesHeadlineSweep returns true if r is the headline-sweep cell for its
// filter family — eps=0.01 for SODA/Bloom, K=14 for Truncation/Scan-ARE/
// Greedy+Merge, bpk=10 for Grafite/SNARF, real_bits=8 for SuRFReal.
func matchesHeadlineSweep(r b6Row) bool {
	switch r.SweepName {
	case "eps":
		return floatNear(r.SweepParam, b6HeadlineEps)
	case "K":
		return floatNear(r.SweepParam, b6HeadlineK)
	case "bpk":
		return floatNear(r.SweepParam, b6HeadlineBPK)
	case "real_bits":
		return floatNear(r.SweepParam, b6HeadlineRealBit)
	}
	return false
}

func floatNear(a, b float64) bool {
	return math.Abs(a-b) <= 1e-9*math.Max(1.0, math.Abs(b))
}

// TestB6Plots regenerates SVGs from bench_results/data/b6_latency_N*/.
// Each per-N directory holds one JSON per filter plus a `_meta.json`
// header; the plotter walks the directory and aggregates rows. Files
// matching the legacy single-file pattern `b6_latency_N*.json` are
// also picked up as a back-compat read-only fallback for unmigrated
// data.
//
// Set PLOT_ONLY=1 (or B6_PLOT=1) to avoid running the heavy measurement
// test. One SVG per (metric, distribution); filters are series within.
func TestB6Plots(t *testing.T) {
	if os.Getenv("PLOT_ONLY") == "" && os.Getenv("B6_PLOT") == "" {
		t.Skip("set B6_PLOT=1 (or PLOT_ONLY=1) to render b6 plots from existing JSON")
	}

	sources, err := discoverB6PlotSources("../bench_results/data")
	if err != nil {
		t.Fatalf("discover b6 plot sources: %v", err)
	}
	if len(sources) == 0 {
		t.Fatalf("no b6 cache found under ../bench_results/data — run TestB6IndustryLatency first")
	}

	for _, src := range sources {
		src := src
		t.Run(src.label, func(t *testing.T) {
			doc, err := loadB6PlotSource(src)
			if err != nil {
				t.Fatalf("load %s: %v", src.label, err)
			}
			if len(doc.Rows) == 0 {
				t.Skipf("no rows in %s", src.label)
			}
			renderB6Plots(t, doc, plotsRootFromSource(src, doc))
		})
	}
}

// plotsRootFromSource derives the per-source plots directory from the
// cache origin. The dir name `b6_latency_N{N}[_<suffix>]` maps to plot
// root `bench_results/plots/b6_N{N}[_<suffix>]`. Legacy file sources
// fall back to the bare `b6_N{N}` form.
func plotsRootFromSource(src b6PlotSource, doc b6Doc) string {
	if src.dir != "" {
		base := filepath.Base(src.dir)
		// strip the "b6_latency_" prefix; what remains is "N{N}" or
		// "N{N}_<suffix>" — use it directly as the plot subdir tag.
		if strings.HasPrefix(base, "b6_latency_") {
			tag := strings.TrimPrefix(base, "b6_latency_")
			return fmt.Sprintf("../bench_results/plots/b6_%s", tag)
		}
	}
	return fmt.Sprintf("../bench_results/plots/b6_N%d", doc.NKeys)
}

// b6PlotSource is one renderable cache origin: either a per-N directory
// (new format: dir + per-filter JSON files + _meta.json) or a legacy
// single-N JSON file kept for back-compat reading.
type b6PlotSource struct {
	label string // human-readable label used as subtest name
	dir   string // populated for directory sources; empty for legacy file
	file  string // populated for legacy single-file sources; empty for dirs
}

// discoverB6PlotSources finds every b6 cache origin under dataDir. It
// prefers the new per-filter directory format `b6_latency_N{N}/` and
// silently skips a legacy `b6_latency_N{N}.json` that has the same N
// as a discovered directory. Otherwise the legacy file is included as
// a read-only fallback.
func discoverB6PlotSources(dataDir string) ([]b6PlotSource, error) {
	var dirs []string
	entries, err := os.ReadDir(dataDir)
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	dirNs := map[string]bool{}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasPrefix(name, "b6_latency_N") {
			continue
		}
		dirs = append(dirs, filepath.Join(dataDir, name))
		dirNs[name+".json"] = true
	}

	jsonPaths, err := filepath.Glob(filepath.Join(dataDir, "b6_latency_N*.json"))
	if err != nil {
		return nil, err
	}
	var legacyFiles []string
	for _, p := range jsonPaths {
		base := filepath.Base(p)
		if dirNs[base] {
			continue // shadowed by a directory of the same N
		}
		legacyFiles = append(legacyFiles, p)
	}
	// Pre-refactor ultra-legacy file (no N suffix). Rare but cheap to support.
	if legacy := filepath.Join(dataDir, "b6_latency.json"); fileExists(legacy) {
		legacyFiles = append(legacyFiles, legacy)
	}

	sort.Strings(dirs)
	sort.Strings(legacyFiles)

	out := make([]b6PlotSource, 0, len(dirs)+len(legacyFiles))
	for _, d := range dirs {
		out = append(out, b6PlotSource{label: filepath.Base(d), dir: d})
	}
	for _, f := range legacyFiles {
		out = append(out, b6PlotSource{label: filepath.Base(f), file: f})
	}
	return out, nil
}

// loadB6PlotSource reads either a per-filter directory or a legacy
// single-file source and returns a unified b6Doc the renderer can
// consume. For the directory case, _meta.json drives nKeys/eps/etc;
// for the legacy case we just unmarshal the doc directly.
func loadB6PlotSource(src b6PlotSource) (b6Doc, error) {
	if src.file != "" {
		buf, err := os.ReadFile(src.file)
		if err != nil {
			return b6Doc{}, err
		}
		var doc b6Doc
		if err := json.Unmarshal(buf, &doc); err != nil {
			return b6Doc{}, err
		}
		return doc, nil
	}

	doc := b6Doc{Type: "b6_latency"}
	// Read _meta.json if present — drives doc.NKeys / Eps / QueryCount
	// used by per-N plot titles.
	if mbuf, err := os.ReadFile(filepath.Join(src.dir, "_meta.json")); err == nil {
		var meta b6MetaDoc
		if err := json.Unmarshal(mbuf, &meta); err == nil {
			doc.NKeys = meta.NKeys
			doc.QueryCount = meta.QueryCount
			doc.QueryStrategy = meta.QueryStrategy
			doc.Eps = meta.Eps
		}
	}

	entries, err := os.ReadDir(src.dir)
	if err != nil {
		return b6Doc{}, err
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if name == "_meta.json" || !strings.HasSuffix(name, ".json") {
			continue
		}
		buf, err := os.ReadFile(filepath.Join(src.dir, name))
		if err != nil {
			return b6Doc{}, fmt.Errorf("read %s: %w", name, err)
		}
		var f b6FilterDoc
		if err := json.Unmarshal(buf, &f); err != nil {
			return b6Doc{}, fmt.Errorf("parse %s: %w", name, err)
		}
		// Fall back to per-filter doc fields if _meta.json was missing
		// (older partial migration / hand-edited cache).
		if doc.NKeys == 0 {
			doc.NKeys = f.NKeys
		}
		if doc.QueryCount == 0 {
			doc.QueryCount = f.QueryCount
		}
		if doc.QueryStrategy == "" {
			doc.QueryStrategy = f.QueryStrategy
		}
		if doc.Eps == 0 {
			doc.Eps = f.Eps
		}
		if doc.Timestamp == "" || f.Timestamp > doc.Timestamp {
			doc.Timestamp = f.Timestamp
			doc.GitCommit = f.GitCommit
		}
		doc.Rows = append(doc.Rows, f.Rows...)
	}
	return doc, nil
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func renderB6Plots(t *testing.T, doc b6Doc, plotsRoot string) {
	// Index rows by (distribution, filter). Per-(metric, dist) plots use
	// only the headline sweep values (one curve per filter through L);
	// cache-pressure and per-L trade-off plots use all sweep values.
	byCell := make(map[struct{ dist, filter string }][]b6Row)
	dists := map[string]struct{}{}
	rangeLensSeen := map[uint64]struct{}{}
	for _, r := range doc.Rows {
		if r.Note != "" {
			continue // skip envelope-rejected / errored cells
		}
		// Skip legacy rows from the eps-based sweep that predate the K-sweep
		// switch for SODA/Scan-ARE/Greedy+Merge/BloomARE. These rows still
		// live in the cache but should not appear on plots — they would
		// duplicate every (L) point at the headline value because
		// matchesHeadlineSweep accepts both "eps" and "K".
		if r.SweepName == "eps" {
			continue
		}
		// Skip rows recorded under non-default parallelism — those belong
		// to cache-contention studies (B6_PARALLEL=N) and would otherwise
		// stack as duplicate points at every L.
		if r.Parallelism > 1 {
			continue
		}
		k := struct{ dist, filter string }{r.Distribution, r.Filter}
		byCell[k] = append(byCell[k], r)
		dists[r.Distribution] = struct{}{}
		rangeLensSeen[r.RangeLen] = struct{}{}
	}

	// Stable distribution order — synthetic first, then SOSD.
	distOrder := []string{
		"clustered", "uniform",
		"sosd_books", "sosd_fb", "sosd_wiki", "sosd_osm",
	}
	finalDists := []string{}
	for _, d := range distOrder {
		if _, ok := dists[d]; ok {
			finalDists = append(finalDists, d)
		}
	}
	for d := range dists {
		seen := false
		for _, fd := range finalDists {
			if fd == d {
				seen = true
				break
			}
		}
		if !seen {
			finalDists = append(finalDists, d)
		}
	}

	// Sorted unique L values — used by per-L trade-off and cache-pressure plots.
	sortedRangeLens := make([]uint64, 0, len(rangeLensSeen))
	for L := range rangeLensSeen {
		sortedRangeLens = append(sortedRangeLens, L)
	}
	sort.Slice(sortedRangeLens, func(i, j int) bool { return sortedRangeLens[i] < sortedRangeLens[j] })

	// Each metric is (subdir, Y label, Y scale, point extractor).
	metrics := []struct {
		subdir  string
		ylabel  string
		yScale  testutils.AxisScale
		yFloor  float64
		yCeil   float64
		extract func(r b6Row) (float64, bool)
	}{
		{
			subdir:  "query_latency",
			ylabel:  "Query Time (ns/op)",
			yScale:  testutils.Log10,
			yCeil:   1000, // BloomARE grows O(L) and goes off-chart; cap at 1 µs
			extract: func(r b6Row) (float64, bool) { return r.QueryNsPerOp, r.QueryNsPerOp > 0 },
		},
		{
			subdir:  "fpr",
			ylabel:  "False Positive Rate (FPR)",
			yScale:  testutils.Log10,
			yFloor:  1.0 / float64(doc.QueryCount), // 1/qc ≈ 3.8e-6
			extract: func(r b6Row) (float64, bool) { return r.FPR, true },
		},
		{
			subdir:  "bpk",
			ylabel:  "Bits per Key (BPK)",
			yScale:  testutils.Linear,
			extract: func(r b6Row) (float64, bool) { return r.BPKUsed, r.BPKUsed > 0 },
		},
		{
			subdir:  "build_throughput",
			ylabel:  "Build Throughput (M keys/sec)",
			yScale:  testutils.Log10,
			extract: func(r b6Row) (float64, bool) { return r.BuildMKeysSec, r.BuildMKeysSec > 0 },
		},
	}

	for _, m := range metrics {
		outDir := filepath.Join(plotsRoot, m.subdir)
		if err := os.MkdirAll(outDir, 0755); err != nil {
			t.Fatalf("mkdir %s: %v", outDir, err)
		}

		for _, dist := range finalDists {
			var ordered []testutils.SeriesData
			if m.subdir == "build_throughput" || m.subdir == "query_latency" {
				// Use FPR-gated selector for both throughput and latency.
				// This ensures we measure performance in the "working regime" (FPR <= eps).
				ordered = buildB6MinFPRMeanSeries(byCell, []string{dist}, doc.Eps, m.extract)
			} else {
				ordered = buildB6PlotSeries(byCell, dist, m.extract)
			}
			if !anyHasPoints(ordered) {
				continue
			}

			svgPath := filepath.Join(outDir, dist+".svg")
			err := testutils.GeneratePerformanceSVG(testutils.PlotConfig{
				Title: fmt.Sprintf("%s — %s (n=2^%d, ε=%.3f)",
					prettyMetric(m.subdir), dist, log2int64(int64(doc.NKeys)), doc.Eps),
				XLabel: "Range Length (L)",
				YLabel: m.ylabel,
				XScale: testutils.Log10,
				YScale: m.yScale,
				YFloor: m.yFloor,
				YCeil:  m.yCeil,
			}, ordered, svgPath)
			if err != nil {
				t.Errorf("svg %s: %v", svgPath, err)
				continue
			}
			fmt.Printf("wrote %s\n", svgPath)
		}

		// Cross-distribution mean — one curve per filter, Y averaged at
		// each L over all distributions where the filter has a data point.
		// Both build_throughput and query_latency use the FPR-gated
		// selector (min-K achieving ε) and exclude the "spread" outlier.
		if m.subdir == "build_throughput" || m.subdir == "query_latency" {
			// Both build_throughput and query_latency means use the FPR-gated
			// selector: each (filter, dist, L) point is taken at the minimum
			// sweep parameter that achieves FPR ≤ ε, not at the fixed headline
			// K=14 which is outside the working regime for many filters (e.g.
			// SODA at K=14 has FPR≈1 on most distributions → trivial 10–13 ns
			// queries that are not representative of real usage).
			//
			// "spread" is also excluded from both means: it is a degenerate
			// case for SODA-family filters (evenly-spaced keys land in trivially
			// few hash buckets, reaching FPR ≤ ε at tiny K with near-zero cost
			// that does not reflect realistic filter construction or query work).
			meanDists := make([]string, 0, len(finalDists))
			for _, d := range finalDists {
				if d != "spread" {
					meanDists = append(meanDists, d)
				}
			}
			ordered := buildB6MinFPRMeanSeries(byCell, meanDists, doc.Eps, m.extract)
			if anyHasPoints(ordered) {
				svgPath := filepath.Join(outDir, "_mean.svg")
				err := testutils.GeneratePerformanceSVG(testutils.PlotConfig{
					Title: fmt.Sprintf("%s — mean across %d distributions (n=2^%d, ε=%.3f)",
						prettyMetric(m.subdir), len(meanDists), log2int64(int64(doc.NKeys)), doc.Eps),
					XLabel: "Range Length (L)",
					YLabel: m.ylabel,
					XScale: testutils.Log10,
					YScale: m.yScale,
					YFloor: m.yFloor,
					YCeil:  m.yCeil,
				}, ordered, svgPath)
				if err != nil {
					t.Errorf("svg %s: %v", svgPath, err)
				} else {
					fmt.Printf("wrote %s\n", svgPath)
				}
			}
		}
	}

	// L-trajectory trade-off plots — one SVG per distribution, each
	// filter is a curve whose points are (BPK_at_L, FPR_at_L) for L in
	// the rangeLens grid at the headline sweep value. Reuses
	// GenerateTradeoffSVG so visual identity matches the FPR-vs-BPK
	// headline plots from comparison_test.go.
	tradeoffDir := filepath.Join(plotsRoot, "tradeoff")
	if err := os.MkdirAll(tradeoffDir, 0755); err != nil {
		t.Fatalf("mkdir %s: %v", tradeoffDir, err)
	}
	yFloor := 1.0 / float64(doc.QueryCount)
	for _, dist := range finalDists {
		ordered := buildB6TradeoffSeries(byCell, dist)
		if !anyHasPoints(ordered) {
			continue
		}
		svgPath := filepath.Join(tradeoffDir, dist+".svg")
		title := fmt.Sprintf("FPR vs BPK trajectory across L — %s (n=2^%d, ε=%.3f)",
			dist, log2int64(int64(doc.NKeys)), doc.Eps)
		if err := testutils.GenerateTradeoffSVG(
			title, "Bits per Key (BPK)", "False Positive Rate (FPR)",
			ordered, svgPath, yFloor,
		); err != nil {
			t.Errorf("svg %s: %v", svgPath, err)
			continue
		}
		fmt.Printf("wrote %s\n", svgPath)
	}

	// Per-L trade-off plots — one SVG per (distribution, L). Each filter
	// is a curve whose points are (BPK_at_sweep, FPR_at_sweep) traced
	// through the K-sweep grid, giving genuine FPR-vs-BPK curves at
	// fixed L.
	for _, dist := range finalDists {
		perLDir := filepath.Join(plotsRoot, "tradeoff_per_L", dist)
		if err := os.MkdirAll(perLDir, 0755); err != nil {
			t.Fatalf("mkdir %s: %v", perLDir, err)
		}
		for _, L := range sortedRangeLens {
			ordered := buildB6TradeoffPerLSeries(byCell, dist, L)
			if !anyHasPoints(ordered) {
				continue
			}
			svgPath := filepath.Join(perLDir, fmt.Sprintf("L%d.svg", L))
			title := fmt.Sprintf("FPR vs BPK (K-sweep) — %s, L=%d (n=2^%d)",
				dist, L, log2int64(int64(doc.NKeys)))
			if err := testutils.GenerateTradeoffSVG(
				title, "Bits per Key (BPK)", "False Positive Rate (FPR)",
				ordered, svgPath, yFloor,
			); err != nil {
				t.Errorf("svg %s: %v", svgPath, err)
				continue
			}
			fmt.Printf("wrote %s\n", svgPath)
		}
	}

	// Cache-pressure plots — one SVG per (distribution, L). X = BPK
	// (linear, capped at 25), Y = query latency (ns, log10). Each
	// filter is a curve through its K-sweep grid; as the filter
	// outgrows L1/L2/L3 caches the curve should kink upward.
	for _, dist := range finalDists {
		cacheDir := filepath.Join(plotsRoot, "cache_pressure", dist)
		if err := os.MkdirAll(cacheDir, 0755); err != nil {
			t.Fatalf("mkdir %s: %v", cacheDir, err)
		}
		for _, L := range sortedRangeLens {
			ordered := buildB6CachePressureSeries(byCell, dist, L)
			if !anyHasPoints(ordered) {
				continue
			}
			svgPath := filepath.Join(cacheDir, fmt.Sprintf("L%d.svg", L))
			title := fmt.Sprintf("Query latency vs filter footprint (cache-pressure) — %s, L=%d", dist, L)
			err := testutils.GeneratePerformanceSVG(testutils.PlotConfig{
				Title:  title,
				XLabel: "Bits per Key (BPK)",
				YLabel: "Query Time (ns/op)",
				XScale: testutils.Linear,
				YScale: testutils.Log10,
				XMax:   25,
			}, ordered, svgPath)
			if err != nil {
				t.Errorf("svg %s: %v", svgPath, err)
				continue
			}
			fmt.Printf("wrote %s\n", svgPath)
		}
	}
}

// buildB6TradeoffSeries builds, per filter, a curve of (BPK_at_L, FPR_at_L)
// points sorted by L (= sorted by BPK in practice for filters whose
// footprint scales monotonically with L). Only headline-sweep rows are
// used so each filter renders as one curve through L.
func buildB6TradeoffSeries(
	byCell map[struct{ dist, filter string }][]b6Row,
	dist string,
) []testutils.SeriesData {
	out := make([]testutils.SeriesData, 0, len(b6PlotOrder))
	for _, fname := range b6PlotOrder {
		rows := collectHeadlineRows(byCell, dist, fname)
		if len(rows) == 0 {
			continue
		}
		sort.Slice(rows, func(i, j int) bool { return rows[i].RangeLen < rows[j].RangeLen })

		s := newB6Series(fname)
		for _, r := range rows {
			if r.BPKUsed <= 0 {
				continue
			}
			s.Points = append(s.Points, testutils.Point{X: r.BPKUsed, Y: floorFPR(r.FPR)})
		}
		if len(s.Points) > 0 {
			out = append(out, s)
		}
	}
	return out
}

// collectHeadlineRows returns rows for (dist, fname) whose sweep matches the
// headline value. For the special "SuRF" key the headline picks the
// (SuRFReal, real_bits=8) representative since SuRF has no single tunable.
//
// byCell is already filtered for legacy/parallelism noise at index time —
// see renderB6Plots — so callers only need to apply the headline match.
func collectHeadlineRows(
	byCell map[struct{ dist, filter string }][]b6Row,
	dist, fname string,
) []b6Row {
	srcNames := []string{fname}
	if fname == "SuRF" {
		srcNames = []string{"SuRFReal"}
	}
	var rows []b6Row
	for _, src := range srcNames {
		for _, r := range byCell[struct{ dist, filter string }{dist, src}] {
			if matchesHeadlineSweep(r) {
				rows = append(rows, r)
			}
		}
	}
	return rows
}

// buildB6TradeoffPerLSeries builds, per filter and fixed L, a curve of
// (BPK, FPR) points traced through the K-sweep grid. Points are sorted
// by BPK so the curve renders monotonically left-to-right. The "SuRF"
// slot folds all three structural variants into a single marker-only
// (NoLine) series since the parameter space is 2D.
func buildB6TradeoffPerLSeries(
	byCell map[struct{ dist, filter string }][]b6Row,
	dist string,
	L uint64,
) []testutils.SeriesData {
	out := make([]testutils.SeriesData, 0, len(b6PlotOrder))
	for _, fname := range b6PlotOrder {
		rows := collectAtL(byCell, dist, fname, L, false)
		if len(rows) == 0 {
			continue
		}
		sort.Slice(rows, func(i, j int) bool { return rows[i].BPKUsed < rows[j].BPKUsed })

		s := newB6Series(fname)
		if fname == "SuRF" {
			s.NoLine = true
		}
		for _, r := range rows {
			if r.BPKUsed <= 0 {
				continue
			}
			s.Points = append(s.Points, testutils.Point{X: r.BPKUsed, Y: floorFPR(r.FPR)})
		}
		if len(s.Points) > 0 {
			out = append(out, s)
		}
	}
	return out
}

// buildB6CachePressureSeries builds, per filter and fixed L, a curve of
// (BPK, query_ns) points traced through the K-sweep grid. Points are
// sorted by BPK. SuRF is rendered as a marker-only point cloud (see
// buildB6TradeoffPerLSeries).
func buildB6CachePressureSeries(
	byCell map[struct{ dist, filter string }][]b6Row,
	dist string,
	L uint64,
) []testutils.SeriesData {
	out := make([]testutils.SeriesData, 0, len(b6PlotOrder))
	for _, fname := range b6PlotOrder {
		rows := collectAtL(byCell, dist, fname, L, true)
		if len(rows) == 0 {
			continue
		}
		sort.Slice(rows, func(i, j int) bool { return rows[i].BPKUsed < rows[j].BPKUsed })

		s := newB6Series(fname)
		if fname == "SuRF" {
			s.NoLine = true
		}
		for _, r := range rows {
			s.Points = append(s.Points, testutils.Point{X: r.BPKUsed, Y: r.QueryNsPerOp})
		}
		if len(s.Points) > 0 {
			out = append(out, s)
		}
	}
	return out
}

// collectAtL gathers rows at the given L for fname (or the full SuRF
// family when fname == "SuRF"). When requireQueryNs is true, rows must
// also have a positive QueryNsPerOp and BPKUsed (cache-pressure filter).
//
// byCell is already filtered for legacy/parallelism noise at index time
// (see renderB6Plots).
func collectAtL(
	byCell map[struct{ dist, filter string }][]b6Row,
	dist, fname string,
	L uint64,
	requireQueryNs bool,
) []b6Row {
	srcNames := []string{fname}
	if fname == "SuRF" {
		srcNames = []string{"SuRFNone", "SuRFHash", "SuRFReal"}
	}
	var rows []b6Row
	for _, src := range srcNames {
		for _, r := range byCell[struct{ dist, filter string }{dist, src}] {
			if r.RangeLen != L {
				continue
			}
			if requireQueryNs && (r.QueryNsPerOp <= 0 || r.BPKUsed <= 0) {
				continue
			}
			rows = append(rows, r)
		}
	}
	return rows
}

// buildB6PlotSeries builds a per-(metric, dist) curve for each filter,
// using only headline-sweep rows so each filter renders as one curve
// through L. The "SuRF" slot uses (SuRFReal, real_bits=8) as the
// single representative since SuRF has no headline tunable.
func buildB6PlotSeries(
	byCell map[struct{ dist, filter string }][]b6Row,
	dist string,
	extract func(r b6Row) (float64, bool),
) []testutils.SeriesData {
	out := make([]testutils.SeriesData, 0, len(b6PlotOrder))
	for _, fname := range b6PlotOrder {
		rows := collectHeadlineRows(byCell, dist, fname)
		if len(rows) == 0 {
			continue
		}
		sort.Slice(rows, func(i, j int) bool { return rows[i].RangeLen < rows[j].RangeLen })

		s := newB6Series(fname)
		for _, r := range rows {
			y, ok := extract(r)
			if !ok {
				continue
			}
			s.Points = append(s.Points, testutils.Point{X: float64(r.RangeLen), Y: y})
		}
		if len(s.Points) > 0 {
			out = append(out, s)
		}
	}
	return out
}

// buildB6MeanSeries builds, per filter, a curve where Y at each L is the
// arithmetic mean of `extract(r)` across all distributions in `dists` that
// have a headline-sweep row at L. Filters with no points across any
// distribution are dropped.
func buildB6MeanSeries(
	byCell map[struct{ dist, filter string }][]b6Row,
	dists []string,
	extract func(r b6Row) (float64, bool),
) []testutils.SeriesData {
	out := make([]testutils.SeriesData, 0, len(b6PlotOrder))
	for _, fname := range b6PlotOrder {
		// Aggregate by L across all dists.
		sumByL := map[uint64]float64{}
		cntByL := map[uint64]int{}
		for _, dist := range dists {
			rows := collectHeadlineRows(byCell, dist, fname)
			for _, r := range rows {
				y, ok := extract(r)
				if !ok {
					continue
				}
				sumByL[r.RangeLen] += y
				cntByL[r.RangeLen]++
			}
		}
		if len(sumByL) == 0 {
			continue
		}
		Ls := make([]uint64, 0, len(sumByL))
		for L := range sumByL {
			Ls = append(Ls, L)
		}
		sort.Slice(Ls, func(i, j int) bool { return Ls[i] < Ls[j] })

		s := newB6Series(fname)
		for _, L := range Ls {
			s.Points = append(s.Points, testutils.Point{
				X: float64(L),
				Y: sumByL[L] / float64(cntByL[L]),
			})
		}
		if len(s.Points) > 0 {
			out = append(out, s)
		}
	}
	return out
}

// collectMinFPRRows returns, for each range length L, the single row with
// the minimum sweep parameter (K / bpk / real_bits) at which the filter
// achieves FPR ≤ targetFPR. Used for build_throughput plots so throughput
// is reported at the cheapest configuration that satisfies the quality bar,
// not at a fixed sweep value that may be completely outside the working regime.
func collectMinFPRRows(
	byCell map[struct{ dist, filter string }][]b6Row,
	dist, fname string,
	targetFPR float64,
) []b6Row {
	srcNames := []string{fname}
	if fname == "SuRF" {
		srcNames = []string{"SuRFReal"}
	}
	byL := map[uint64][]b6Row{}
	for _, src := range srcNames {
		for _, r := range byCell[struct{ dist, filter string }{dist, src}] {
			if r.FPR <= targetFPR {
				byL[r.RangeLen] = append(byL[r.RangeLen], r)
			}
		}
	}
	result := make([]b6Row, 0, len(byL))
	for _, rows := range byL {
		sort.Slice(rows, func(i, j int) bool {
			return rows[i].SweepParam < rows[j].SweepParam
		})
		result = append(result, rows[0])
	}
	return result
}

// buildB6MinFPRMeanSeries is the cross-distribution mean (and per-dist) variant
// of the FPR-gated series builder, generalised to any metric via an extractor.
// For each (filter, L) it averages extract(r) over all dists that have at
// least one row passing collectMinFPRRows at targetFPR. Distributions where
// the filter never reaches targetFPR are silently omitted from the average.
//
// Coverage guard: a point at L is only emitted when the number of
// contributing distributions is strictly greater than half the filter's
// maximum coverage across all L values. This prevents artificial "improvement"
// artefacts when difficult distributions gradually drop out of the average
// (e.g. Grafite on clustered data failing at large L, causing the mean to
// plunge from 4 K ns to 85 ns once that distribution disappears).
// The threshold is per-filter so filters with inherently limited distribution
// support (e.g. SNARF, SuRF) are not penalised.
func buildB6MinFPRMeanSeries(
	byCell map[struct{ dist, filter string }][]b6Row,
	dists []string,
	targetFPR float64,
	extract func(r b6Row) (float64, bool),
) []testutils.SeriesData {
	out := make([]testutils.SeriesData, 0, len(b6PlotOrder))
	for _, fname := range b6PlotOrder {
		sumByL := map[uint64]float64{}
		cntByL := map[uint64]int{}
		for _, dist := range dists {
			for _, r := range collectMinFPRRows(byCell, dist, fname, targetFPR) {
				y, ok := extract(r)
				if !ok {
					continue
				}
				sumByL[r.RangeLen] += y
				cntByL[r.RangeLen]++
			}
		}
		if len(sumByL) == 0 {
			continue
		}
		// Determine this filter's maximum distribution coverage across all L.
		maxCnt := 0
		for _, c := range cntByL {
			if c > maxCnt {
				maxCnt = c
			}
		}
		Ls := make([]uint64, 0, len(sumByL))
		for L := range sumByL {
			Ls = append(Ls, L)
		}
		sort.Slice(Ls, func(i, j int) bool { return Ls[i] < Ls[j] })
		s := newB6Series(fname)
		for _, L := range Ls {
			// Skip L where coverage has dropped to ≤ half of this filter's
			// maximum — the average would be over a qualitatively different
			// (easier) subset and no longer comparable across L values.
			if cntByL[L]*2 <= maxCnt {
				continue
			}
			s.Points = append(s.Points, testutils.Point{
				X: float64(L),
				Y: sumByL[L] / float64(cntByL[L]),
			})
		}
		if len(s.Points) > 0 {
			out = append(out, s)
		}
	}
	return out
}

func newB6Series(fname string) testutils.SeriesData {
	style, ok := b6SeriesStyles[fname]
	if !ok {
		style = SeriesStyle{Name: fname, Color: "#9ca3af", Marker: "circle"}
	}
	return testutils.SeriesData{
		Name:   style.Name,
		Color:  style.Color,
		Marker: style.Marker,
		Dashed: style.Dashed,
	}
}

// floorFPR pins a 0-FPR observation just below the YFloor so the log10
// axis renders it on the floor line; the shared "0 FP" marker row in
// GenerateTradeoffSVG conveys the real meaning.
func floorFPR(fpr float64) float64 {
	if fpr <= 0 {
		return 1e-300
	}
	return fpr
}

func anyHasPoints(series []testutils.SeriesData) bool {
	for _, s := range series {
		if len(s.Points) > 0 {
			return true
		}
	}
	return false
}

func prettyMetric(subdir string) string {
	switch subdir {
	case "query_latency":
		return "Query Latency vs L"
	case "fpr":
		return "FPR vs L"
	case "bpk":
		return "BPK vs L"
	case "build_throughput":
		return "Build Throughput vs L"
	default:
		return subdir
	}
}

func log2int64(n int64) int {
	k := 0
	for n > 1 {
		n >>= 1
		k++
	}
	return k
}
