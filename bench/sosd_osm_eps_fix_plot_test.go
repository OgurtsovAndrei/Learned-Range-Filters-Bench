package bench_test

import (
	"path/filepath"
	"testing"

	"Thesis/testutils"
)

// TestEpsFix_OSM_FullPlot writes an FPR-vs-BPK trade-off SVG from the data
// produced by TestEpsFix_HybridScan_OSM_Full on n=800M, L=128, smart queries
// (50% near-key + 30% in-gap + 20% uniform). Points are precomputed so the
// plot can be regenerated cheaply without rerunning the 15-min sweep.
//
// Run: go test -v -run TestEpsFix_OSM_FullPlot ./bench/
func TestEpsFix_OSM_FullPlot(t *testing.T) {
	yFloor := 1.0 / 30000.0 // 30K queries → measurement floor 3.3e-5

	type pt struct {
		bpk float64
		fpr float64
	}
	// Floor 0-FPR rows to yFloor so log-scale plot renders them at the floor
	// line instead of dropping the point. The floor is annotated on the SVG.
	clamp := func(p pt) testutils.Point {
		y := p.fpr
		if y < yFloor {
			y = yFloor
		}
		return testutils.Point{X: p.bpk, Y: y}
	}

	style := func(name string) SeriesStyle {
		return DefaultSeriesStyles[name]
	}
	mkSeries := func(name string, pts []pt) testutils.SeriesData {
		s := style(name)
		out := testutils.SeriesData{
			Name:  s.Name,
			Color: s.Color,
			Marker: func() string {
				if s.Marker == "" {
					return "circle"
				}
				return s.Marker
			}(),
			Dashed: s.Dashed,
		}
		for _, p := range pts {
			out.Points = append(out.Points, clamp(p))
		}
		return out
	}

	// K∈{40,44,48,52} on OSM 800M, L=128, smart queries 30K
	// Numbers from /tmp/osm_full_local_K.log — per-cluster K_local fix
	// (cf1f685 + localK rescaling).
	series := []testutils.SeriesData{
		mkSeries("Scan-ARE-Trunc", []pt{
			{12.422, 1.74033e-1},
			{16.728, 1.94667e-2},
			{20.735, 9.33333e-4},
			{24.785, 3.33333e-5},
		}),
		mkSeries("Scan-ARE-SODA", []pt{
			{12.728, 1.43967e-1},
			{16.742, 8.50000e-3},
			{20.735, 6.00000e-4},
			{24.785, 0},
		}),
		mkSeries("Greedy+Merge-Trunc", []pt{
			{12.891, 7.26000e-2},
			{16.750, 4.70000e-3},
			{20.719, 1.33333e-4},
			{24.707, 0},
		}),
		mkSeries("Greedy+Merge-SODA", []pt{
			{12.901, 1.38433e-1},
			{16.750, 1.04667e-2},
			{20.719, 4.33333e-4},
			{24.707, 0},
		}),
		mkSeries("SODA", []pt{
			{12.662, 8.49000e-2},
			{16.670, 6.16667e-3},
			{20.671, 4.00000e-4},
			{24.671, 0},
		}),
		mkSeries("Grafite", []pt{
			{13.066, 8.41333e-2},
			{17.074, 5.23333e-3},
			{21.074, 2.66667e-4},
			{25.074, 3.33333e-5},
		}),
	}

	out := filepath.Join("..", "bench_results", "plots", "osm_800M_eps_fix_tradeoff.svg")
	title := "FPR vs BPK — sosd_osm n=800M, L=128 (K∈{40,44,48,52}, per-cluster K_local)"
	cfg := testutils.PlotConfig{
		Title:  title,
		XLabel: "Bits per Key (BPK)",
		YLabel: "False Positive Rate (FPR)",
		XScale: testutils.Linear,
		YScale: testutils.Log10,
		YFloor: yFloor,
		XMax:   28,
	}
	if err := testutils.GeneratePerformanceSVG(cfg, series, out); err != nil {
		t.Fatalf("svg: %v", err)
	}
	t.Logf("wrote %s", out)
}
