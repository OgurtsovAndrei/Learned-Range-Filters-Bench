package bench_test

import (
	"Thesis/testutils"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// b6SeriesStyles extends DefaultSeriesStyles with the one filter that B6
// shows but the FPR-vs-BPK headline does not (Truncation). The headline
// scheme reserves "warm + circle" for the this-work line; Truncation is
// from chapter 4 (intermediate building block) so it gets a fourth warm
// hue (crimson) keeping the same circle marker for family coherence.
//
// Filters not listed here fall through to a default gray rendering.
var b6SeriesStyles = func() map[string]SeriesStyle {
	m := make(map[string]SeriesStyle, len(DefaultSeriesStyles)+2)
	for k, v := range DefaultSeriesStyles {
		m[k] = v
	}
	m["Truncation"] = SeriesStyle{Name: "Truncation", Color: "#b91c1c", Marker: "circle"}
	// SuRF is one family rendered as a marker-only point cloud across all
	// three structural variants (None / Hash / Real). Inherit the
	// SuRFReal(8) palette so plots stay consistent with comparison_test.go.
	surfColor := "#0f172a"
	surfMarker := "diamond"
	if s, ok := DefaultSeriesStyles["SuRFReal(8)"]; ok {
		surfColor = s.Color
		surfMarker = s.Marker
	}
	m["SuRF"] = SeriesStyle{Name: "SuRF", Color: surfColor, Marker: surfMarker}
	// Keep SuRFReal entry as a fallback used only by the per-(metric, dist)
	// plots that pick a single representative cell; same palette as SuRF.
	m["SuRFReal"] = SeriesStyle{Name: "SuRFReal", Color: surfColor, Marker: surfMarker}
	return m
}()

// b6 series rendering order — matches the headline plot ordering with
// Truncation inserted into the "this work" warm cluster, between SODA
// (the academic predecessor) and Scan-ARE / Greedy+Merge (the headline
// pair that build on it).
var b6PlotOrder = []string{
	"Grafite",
	"SNARF",
	"SuRF",
	"SODA",
	"Truncation",
	"Scan-ARE",
	"Greedy+Merge",
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

// TestB6Plots regenerates SVGs from bench_results/data/b6_latency.json.
// Set PLOT_ONLY=1 (or B6_PLOT=1) to avoid running the heavy measurement
// test. One SVG per (metric, distribution); filters are series within.
func TestB6Plots(t *testing.T) {
	if os.Getenv("PLOT_ONLY") == "" && os.Getenv("B6_PLOT") == "" {
		t.Skip("set B6_PLOT=1 (or PLOT_ONLY=1) to render b6 plots from existing JSON")
	}

	jsonPath := "../bench_results/data/b6_latency.json"
	buf, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("read %s: %v", jsonPath, err)
	}
	var doc b6Doc
	if err := json.Unmarshal(buf, &doc); err != nil {
		t.Fatalf("parse %s: %v", jsonPath, err)
	}
	if len(doc.Rows) == 0 {
		t.Fatalf("no rows in %s — run TestB6IndustryLatency first", jsonPath)
	}

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
		k := struct{ dist, filter string }{r.Distribution, r.Filter}
		byCell[k] = append(byCell[k], r)
		dists[r.Distribution] = struct{}{}
		rangeLensSeen[r.RangeLen] = struct{}{}
	}

	// Stable distribution order — synthetic first, then SOSD.
	distOrder := []string{
		"clustered", "uniform", "spread",
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
		extract func(r b6Row) (float64, bool)
	}{
		{
			subdir:  "query_latency",
			ylabel:  "Query Time (ns/op)",
			yScale:  testutils.Log10,
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

	plotsRoot := fmt.Sprintf("../bench_results/plots/b6_N%d", doc.NKeys)
	for _, m := range metrics {
		outDir := filepath.Join(plotsRoot, m.subdir)
		if err := os.MkdirAll(outDir, 0755); err != nil {
			t.Fatalf("mkdir %s: %v", outDir, err)
		}

		for _, dist := range finalDists {
			ordered := buildB6PlotSeries(byCell, dist, m.extract)
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
			}, ordered, svgPath)
			if err != nil {
				t.Errorf("svg %s: %v", svgPath, err)
				continue
			}
			fmt.Printf("wrote %s\n", svgPath)
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
