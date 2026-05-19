package bench_test

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// paperDistName maps the internal distribution identifier used by the
// runner to a reader-facing label suitable for a paper figure caption.
// Unknown ids are returned untouched.
func paperDistName(dist string) string {
	switch dist {
	case "sosd_fb":
		return "Facebook user IDs"
	case "sosd_wiki":
		return "Wikipedia timestamps"
	case "sosd_osm":
		return "OSM cell IDs"
	case "sosd_books":
		return "Amazon books (200M)"
	case "sosd_books_800m":
		return "Amazon books (800M)"
	case "uniform":
		return "uniform"
	case "clustered":
		return "clustered"
	default:
		return dist
	}
}

// paperN renders n as 2²⁰ etc. using Unicode super-script digits so the
// title line reads as a typeset formula instead of "n=2^20".
func paperN(log2N int) string {
	return "2" + superscriptIntForPaper(log2N)
}

// superscriptIntForPaper mirrors the testutils package-private helper but
// is kept local so we don't have to export it.
func superscriptIntForPaper(e int) string {
	digits := []rune{'⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'}
	neg := e < 0
	if neg {
		e = -e
	}
	if e == 0 {
		return "⁰"
	}
	var rs []rune
	for e > 0 {
		rs = append([]rune{digits[e%10]}, rs...)
		e /= 10
	}
	if neg {
		rs = append([]rune{'⁻'}, rs...)
	}
	return string(rs)
}

// b6PaperPlotOrder is the cleaned-down series list for paper figures:
// industry baselines + our SegARE only. Same family-style mapping as the
// full plotter (b6SeriesStyles), so colours/markers stay consistent.
var b6PaperPlotOrder = []string{
	"Grafite",
	"SNARF",
	"SuRF",
	"Rosetta",
	"SegARE",
}

// TestB6PaperPlots renders the paper-figure variant: only Grafite/SNARF/SuRF/
// Rosetta + SegARE, written under bench_results/plots_paper/. The rendering
// pipeline is reused verbatim — b6PlotOrder is swapped under a defer so the
// regular TestB6Plots is unaffected.
//
//	B6_PAPER=1 go test -v -run TestB6PaperPlots ./bench/
func TestB6PaperPlots(t *testing.T) {
	if os.Getenv("B6_PAPER") == "" && os.Getenv("PLOT_ONLY") == "" && os.Getenv("B6_PLOT") == "" {
		t.Skip("set B6_PAPER=1 (or B6_PLOT=1 / PLOT_ONLY=1) to render the paper-variant b6 plots")
	}

	sources, err := discoverB6PlotSources("../bench_results/data")
	if err != nil {
		t.Fatalf("discover b6 plot sources: %v", err)
	}
	if len(sources) == 0 {
		t.Fatalf("no b6 cache found under ../bench_results/data — run TestB6IndustryLatency first")
	}

	savedOrder := b6PlotOrder
	b6PlotOrder = b6PaperPlotOrder
	t.Cleanup(func() { b6PlotOrder = savedOrder })

	// Presentation palette: each series gets a unique hue so the five
	// curves stay visually distinct on a slide / printed page. Default
	// styling had Grafite teal and SegARE light-green, which collided
	// with Rosetta dark-green — visible legend collapse on dense plots.
	// Exact palette from Thesis/text/defence/slides/defence.tex so paper
	// figures stay visually identical to the defence deck.
	paperStyles := map[string]SeriesStyle{
		"Grafite": {Name: "Grafite", Color: "#0F766E", Marker: "diamond"}, // teal
		"SNARF":   {Name: "SNARF", Color: "#1E3A8A", Marker: "diamond"},   // deep blue
		"SuRF":    {Name: "SuRF", Color: "#DC2626", Marker: "diamond"},    // red
		"Rosetta": {Name: "Rosetta", Color: "#15803D", Marker: "diamond"}, // dark green
		"SegARE":  {Name: "SegARE", Color: "#D946EF", Marker: "circle"},   // magenta (was Scan-ARE on slides)
		// SegARE-InGapFPR — sibling, distinct hue (orange) so the two
		// policy variants don't blur together on the page.
		"SegARE-InGapFPR": {Name: "SegARE-InGapFPR", Color: "#EA580C", Marker: "square"}, // orange accent (thAccent2 in defence.tex)
	}
	savedStyles := make(map[string]SeriesStyle, len(paperStyles))
	for name, st := range paperStyles {
		savedStyles[name] = b6SeriesStyles[name]
		b6SeriesStyles[name] = st
	}
	t.Cleanup(func() {
		for name, st := range savedStyles {
			b6SeriesStyles[name] = st
		}
	})

	// Paper-figure titles: human-readable distribution names, Unicode
	// exponents for n, no "(K-sweep)" jargon. Defence titles are restored
	// on cleanup so a follow-up TestB6Plots run gets the original strings.
	savedTitles := b6Titles
	b6Titles = b6Titler{
		Metric: func(metric, dist string, log2N int, eps float64) string {
			return fmt.Sprintf("%s on %s (n = %s, ε = %.2f)",
				metric, paperDistName(dist), paperN(log2N), eps)
		},
		MetricMean: func(metric string, nDists, log2N int, eps float64) string {
			return fmt.Sprintf("%s, mean over %d distributions (n = %s, ε = %.2f)",
				metric, nDists, paperN(log2N), eps)
		},
		Tradeoff: func(dist string, log2N int, eps float64) string {
			return fmt.Sprintf("False-positive rate vs space on %s (n = %s, ε = %.2f, L sweep)",
				paperDistName(dist), paperN(log2N), eps)
		},
		TradeoffPerL: func(dist string, L uint64, log2N int) string {
			return fmt.Sprintf("False-positive rate vs space on %s (n = %s, L = %d)",
				paperDistName(dist), paperN(log2N), L)
		},
		CachePressure: func(dist string, L uint64) string {
			return fmt.Sprintf("Query latency vs filter footprint on %s (L = %d)",
				paperDistName(dist), L)
		},
	}
	t.Cleanup(func() { b6Titles = savedTitles })

	// Drop non-monotone tradeoff points (parameter-sweep noise) so the
	// paper figures show clean Pareto fronts.
	savedMonotone := b6TradeoffMonotone
	b6TradeoffMonotone = true
	t.Cleanup(func() { b6TradeoffMonotone = savedMonotone })

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
			renderB6Plots(t, doc, paperPlotsRootFromSource(src, doc))
		})
	}
}

// paperPlotsRootFromSource mirrors plotsRootFromSource but writes under
// bench_results/plots_paper/ so paper figures sit beside (not on top of)
// the full plot tree.
func paperPlotsRootFromSource(src b6PlotSource, doc b6Doc) string {
	if src.dir != "" {
		base := filepath.Base(src.dir)
		if strings.HasPrefix(base, "b6_latency_") {
			tag := strings.TrimPrefix(base, "b6_latency_")
			return fmt.Sprintf("../bench_results/plots_paper/b6_%s", tag)
		}
	}
	return fmt.Sprintf("../bench_results/plots_paper/b6_N%d", doc.NKeys)
}
