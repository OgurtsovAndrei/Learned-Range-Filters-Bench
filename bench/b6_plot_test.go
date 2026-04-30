package bench_test

import (
	"Thesis/testutils"
	"encoding/json"
	"fmt"
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
	m := make(map[string]SeriesStyle, len(DefaultSeriesStyles)+1)
	for k, v := range DefaultSeriesStyles {
		m[k] = v
	}
	m["Truncation"] = SeriesStyle{Name: "Truncation", Color: "#b91c1c", Marker: "circle"}
	return m
}()

// b6 series rendering order — matches the headline plot ordering with
// Truncation inserted into the "this work" warm cluster, between SODA
// (the academic predecessor) and Scan-ARE / Greedy+Merge (the headline
// pair that build on it).
var b6PlotOrder = []string{
	"Grafite",
	"SNARF",
	"SuRFReal(8)",
	"SODA",
	"Truncation",
	"Scan-ARE",
	"Greedy+Merge",
	"BloomARE",
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

	// Index rows by (distribution, filter) so we can produce one series
	// per filter inside each distribution-scoped plot.
	byCell := make(map[struct{ dist, filter string }][]b6Row)
	dists := map[string]struct{}{}
	for _, r := range doc.Rows {
		if r.Note != "" {
			continue // skip envelope-rejected / errored cells
		}
		k := struct{ dist, filter string }{r.Distribution, r.Filter}
		byCell[k] = append(byCell[k], r)
		dists[r.Distribution] = struct{}{}
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

	// Each metric is (subdir, Y label, Y scale, point extractor).
	metrics := []struct {
		subdir string
		ylabel string
		yScale testutils.AxisScale
		yFloor float64
		extract func(r b6Row) (float64, bool)
	}{
		{
			subdir: "query_latency",
			ylabel: "Query Time (ns/op)",
			yScale: testutils.Log10,
			extract: func(r b6Row) (float64, bool) { return r.QueryNsPerOp, r.QueryNsPerOp > 0 },
		},
		{
			subdir: "fpr",
			ylabel: "False Positive Rate (FPR)",
			yScale: testutils.Log10,
			yFloor: 1.0 / float64(doc.QueryCount), // 1/qc ≈ 3.8e-6
			extract: func(r b6Row) (float64, bool) { return r.FPR, true },
		},
		{
			subdir: "bpk",
			ylabel: "Bits per Key (BPK)",
			yScale: testutils.Linear,
			extract: func(r b6Row) (float64, bool) { return r.BPKUsed, r.BPKUsed > 0 },
		},
		{
			subdir: "build_throughput",
			ylabel: "Build Throughput (M keys/sec)",
			yScale: testutils.Log10,
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
}

func buildB6PlotSeries(
	byCell map[struct{ dist, filter string }][]b6Row,
	dist string,
	extract func(r b6Row) (float64, bool),
) []testutils.SeriesData {
	out := make([]testutils.SeriesData, 0, len(b6PlotOrder))
	for _, fname := range b6PlotOrder {
		rows := byCell[struct{ dist, filter string }{dist, fname}]
		if len(rows) == 0 {
			continue
		}
		sort.Slice(rows, func(i, j int) bool { return rows[i].RangeLen < rows[j].RangeLen })

		style, ok := b6SeriesStyles[fname]
		if !ok {
			style = SeriesStyle{Name: fname, Color: "#9ca3af", Marker: "circle"}
		}
		s := testutils.SeriesData{
			Name:   style.Name,
			Color:  style.Color,
			Marker: style.Marker,
			Dashed: style.Dashed,
		}
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
