package datasets_test

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"Thesis-bench-industry/bench/datasets"
	"Thesis/testutils"
)

// movingAverage applies a centred moving-average filter of odd width w to
// the Y values of pts. At edges the window is shrunk symmetrically rather
// than wrapped or zero-padded — keeps endpoint values honest. X values are
// preserved.
func movingAverage(pts []testutils.Point, w int) []testutils.Point {
	if w < 2 {
		out := make([]testutils.Point, len(pts))
		copy(out, pts)
		return out
	}
	half := w / 2
	out := make([]testutils.Point, len(pts))
	for i := range pts {
		lo := i - half
		hi := i + half
		if lo < 0 {
			lo = 0
		}
		if hi >= len(pts) {
			hi = len(pts) - 1
		}
		var sum float64
		for j := lo; j <= hi; j++ {
			sum += pts[j].Y
		}
		out[i] = testutils.Point{X: pts[i].X, Y: sum / float64(hi-lo+1)}
	}
	return out
}

// TestNYCTaxi_RushHourHistograms produces four raw-trip-count histograms
// (no dedup) at increasing time scales: day → week → month → year. Multiple
// trips in the same second are kept, so the histogram peaks reveal rush-hour
// bursts and night valleys. All histograms use BinHistogram (streaming,
// O(nBins) memory).
func TestNYCTaxi_RushHourHistograms(t *testing.T) {
	monthPath := nycTaxiPath("yellow_tripdata_2009-01.parquet")
	if _, err := os.Stat(monthPath); err != nil {
		t.Skipf("2009-01 file not available: %v", err)
	}

	histDir := filepath.Join(filepath.Dir(monthPath), "..", "bench_results",
		"plots", "distributions")
	histDirLinear := filepath.Join(filepath.Dir(monthPath), "..", "bench_results",
		"plots", "distributions_linear")
	histDirSmooth := filepath.Join(filepath.Dir(monthPath), "..", "bench_results",
		"plots", "distributions_smoothed")
	os.MkdirAll(histDir, 0755)
	os.MkdirAll(histDirLinear, 0755)
	os.MkdirAll(histDirSmooth, 0755)
	smoothWindows := []int{5, 7, 11}

	const nBins = 1000

	scales := []struct {
		name      string
		filesGlob string
		lo, hi    time.Time
		xLabel    string
	}{
		{
			name:      "day",
			filesGlob: "yellow_tripdata_2009-01.parquet",
			lo:        time.Date(2009, 1, 15, 0, 0, 0, 0, time.UTC),
			hi:        time.Date(2009, 1, 16, 0, 0, 0, 0, time.UTC),
			xLabel:    "Hour of day (2009-01-15, Thursday)",
		},
		{
			name:      "week",
			filesGlob: "yellow_tripdata_2009-01.parquet",
			lo:        time.Date(2009, 1, 12, 0, 0, 0, 0, time.UTC),
			hi:        time.Date(2009, 1, 19, 0, 0, 0, 0, time.UTC),
			xLabel:    "Day of week (Mon 2009-01-12 → Sun 2009-01-18)",
		},
		{
			name:      "month",
			filesGlob: "yellow_tripdata_2009-01.parquet",
			lo:        time.Date(2009, 1, 1, 0, 0, 0, 0, time.UTC),
			hi:        time.Date(2009, 2, 1, 0, 0, 0, 0, time.UTC),
			xLabel:    "Day of month (January 2009)",
		},
		{
			name:      "year",
			filesGlob: "yellow_tripdata_2009-*.parquet",
			lo:        time.Date(2009, 1, 1, 0, 0, 0, 0, time.UTC),
			hi:        time.Date(2010, 1, 1, 0, 0, 0, 0, time.UTC),
			xLabel:    "Month of year (2009)",
		},
	}

	for _, sc := range scales {
		t.Run(sc.name, func(t *testing.T) {
			matches, _ := filepath.Glob(nycTaxiPath(sc.filesGlob))
			if len(matches) == 0 {
				t.Skipf("no files match %s", sc.filesGlob)
			}
			sort.Strings(matches)

			r := &datasets.NYCTaxiPickupReader{
				Files: matches,
				Label: "nyc_yellow_2009_" + sc.name,
			}
			lo := uint64(sc.lo.UnixNano())
			hi := uint64(sc.hi.UnixNano())

			t0 := time.Now()
			counts, err := r.BinHistogram(lo, hi, nBins)
			if err != nil {
				t.Fatalf("BinHistogram: %v", err)
			}
			elapsed := time.Since(t0)

			var total uint64
			var maxCount uint64
			emptyBins := 0
			for _, c := range counts {
				total += c
				if c > maxCount {
					maxCount = c
				}
				if c == 0 {
					emptyBins++
				}
			}
			fmt.Printf("\n=== %s ===\n", r.Label)
			fmt.Printf("range=[%s, %s]  files=%d  bins=%d  binWidth=%v\n",
				sc.lo.Format("2006-01-02"), sc.hi.Format("2006-01-02"),
				len(matches), nBins, time.Duration((hi-lo)/uint64(nBins)))
			fmt.Printf("trips total=%d  bins-empty=%d  max-bin=%d  read-time=%s\n",
				total, emptyBins, maxCount, elapsed.Round(time.Millisecond))

			ptsLog := make([]testutils.Point, nBins)
			ptsLin := make([]testutils.Point, nBins)
			for i, c := range counts {
				x := float64(i) / float64(nBins)
				yLin := float64(c) / float64(maxCount)
				yLog := yLin
				if yLog == 0 {
					yLog = 1e-9 // log-y can't render 0; sink empty bins
				}
				ptsLog[i] = testutils.Point{X: x, Y: yLog}
				ptsLin[i] = testutils.Point{X: x, Y: yLin}
			}
			title := fmt.Sprintf("Trip Count — %s (n=%d trips, %d bins, raw)",
				r.Label, total, nBins)
			seriesLog := []testutils.SeriesData{{
				Name: r.Label, Color: "#f4b400", Marker: "none", Points: ptsLog,
			}}
			seriesLin := []testutils.SeriesData{{
				Name: r.Label, Color: "#f4b400", Marker: "none", Points: ptsLin,
			}}

			logPath := filepath.Join(histDir,
				fmt.Sprintf("hist_nyc_yellow_2009_%s_raw.svg", sc.name))
			err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
				Title: title, XLabel: sc.xLabel,
				YLabel: "Relative Density (trips per bin / max bin)",
				YScale: testutils.Log10, XMax: 25,
			}, seriesLog, logPath)
			if err != nil {
				t.Fatalf("log svg: %v", err)
			}

			linPath := filepath.Join(histDirLinear,
				fmt.Sprintf("hist_nyc_yellow_2009_%s_raw.svg", sc.name))
			err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
				Title: title, XLabel: sc.xLabel,
				YLabel: "Trips per bin / max bin",
				YScale: testutils.Linear, XMax: 25,
			}, seriesLin, linPath)
			if err != nil {
				t.Fatalf("linear svg: %v", err)
			}
			fmt.Printf("svg → %s\nsvg → %s\n", logPath, linPath)

			for _, w := range smoothWindows {
				smoothed := movingAverage(ptsLin, w)
				path := filepath.Join(histDirSmooth,
					fmt.Sprintf("hist_nyc_yellow_2009_%s_raw_w%d.svg", sc.name, w))
				err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
					Title:  fmt.Sprintf("%s — smoothed (window=%d bins)", title, w),
					XLabel: sc.xLabel,
					YLabel: fmt.Sprintf("Mean of %d-bin window (normalized)", w),
					YScale: testutils.Linear, XMax: 25,
				}, []testutils.SeriesData{{
					Name: r.Label, Color: "#0f9d58", Marker: "none", Points: smoothed,
				}}, path)
				if err != nil {
					t.Fatalf("smoothed svg w=%d: %v", w, err)
				}
				fmt.Printf("svg → %s\n", path)
			}
		})
	}
}
