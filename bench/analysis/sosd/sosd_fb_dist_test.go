package sosd_test

import (
	"Thesis-bench-industry/bench/internal/keygen"
	"Thesis-bench-industry/bench/internal/benchutil"
	"Thesis/testutils"
	"fmt"
	"os"
	"testing"
)

func sosdPath(name string) string {
	return keygen.SOSDPath(name)
}

func loadSOSDUint64(path string, maxKeys int) ([]uint64, error) {
	return keygen.LoadSOSDUint64(path, maxKeys)
}

func loadSOSDUint32(path string, maxKeys int) ([]uint64, error) {
	return keygen.LoadSOSDUint32(path, maxKeys)
}

func histogram(keys []uint64, nBins int) []testutils.Point {
	return benchutil.Histogram(keys, nBins)
}

func normalizedCDF(keys []uint64, sampleEvery int) []testutils.Point {
	return benchutil.NormalizedCDF(keys, sampleEvery)
}



func TestDistribution_SOSD_FB_Histogram(t *testing.T) {

	path := sosdPath("fb_200M_uint64")
	keys, err := loadSOSDUint64(path, 0)
	if err != nil {
		t.Skipf("SOSD fb_200M_uint64 not available: %v", err)
	}

	os.MkdirAll("../bench_results/plots/distributions", 0755)

	// Histogram
	histSeries := []testutils.SeriesData{{
		Name:   "sosd_fb",
		Color:  "#e74c3c",
		Marker: "none",
		Points: histogram(keys, 1000),
	}}
	histPath := "../bench_results/plots/distributions/hist_sosd_fb.svg"
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  "Key Density — sosd_fb (n=200M, 1000 bins)",
		XLabel: "Normalized Key Position",
		YLabel: "Relative Density",
		YScale: testutils.Log10,
		XMax:   25,
	}, histSeries, histPath)
	if err != nil {
		t.Errorf("histogram SVG failed: %v", err)
	} else {
		fmt.Printf("Histogram written to %s\n", histPath)
	}

	// CDF
	cdfSeries := []testutils.SeriesData{{
		Name:   "sosd_fb",
		Color:  "#e74c3c",
		Marker: "none",
		Points: normalizedCDF(keys, 256),
	}}
	cdfPath := "../bench_results/plots/distributions/cdf_sosd_fb.svg"
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("CDF — sosd_fb (n=%d, normalized)", len(keys)),
		XLabel: "Normalized Key Position",
		YLabel: "Cumulative Fraction",
		XMax:   25,
	}, cdfSeries, cdfPath)
	if err != nil {
		t.Errorf("CDF SVG failed: %v", err)
	} else {
		fmt.Printf("CDF written to %s\n", cdfPath)
	}
}

func TestDistribution_SOSD_Wiki_Histogram(t *testing.T) {
	path := sosdPath("wiki_ts_200M_uint64")
	keys, err := loadSOSDUint64(path, 0)
	if err != nil {
		t.Skipf("SOSD wiki_ts_200M_uint64 not available: %v", err)
	}

	os.MkdirAll("../bench_results/plots/distributions", 0755)

	histSeries := []testutils.SeriesData{{
		Name:   "sosd_wiki",
		Color:  "#3498db",
		Marker: "none",
		Points: histogram(keys, 1000),
	}}
	histPath := "../bench_results/plots/distributions/hist_sosd_wiki.svg"
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("Key Density — sosd_wiki (n=%d, 1000 bins)", len(keys)),
		XLabel: "Normalized Key Position",
		YLabel: "Relative Density",
		YScale: testutils.Log10,
		XMax:   25,
	}, histSeries, histPath)
	if err != nil {
		t.Errorf("histogram SVG failed: %v", err)
	} else {
		fmt.Printf("Histogram written to %s\n", histPath)
	}

	cdfSeries := []testutils.SeriesData{{
		Name:   "sosd_wiki",
		Color:  "#3498db",
		Marker: "none",
		Points: normalizedCDF(keys, 256),
	}}
	cdfPath := "../bench_results/plots/distributions/cdf_sosd_wiki.svg"
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("CDF — sosd_wiki (n=%d, normalized)", len(keys)),
		XLabel: "Normalized Key Position",
		YLabel: "Cumulative Fraction",
		XMax:   25,
	}, cdfSeries, cdfPath)
	if err != nil {
		t.Errorf("CDF SVG failed: %v", err)
	} else {
		fmt.Printf("CDF written to %s\n", cdfPath)
	}
}

func TestDistribution_SOSD_OSM_Histogram(t *testing.T) {
	path := sosdPath("osm_cellids_800M_uint64")
	keys, err := loadSOSDUint64(path, 0)
	if err != nil {
		t.Skipf("SOSD osm_cellids_800M_uint64 not available: %v", err)
	}

	os.MkdirAll("../bench_results/plots/distributions", 0755)

	histSeries := []testutils.SeriesData{{
		Name:   "sosd_osm",
		Color:  "#27ae60",
		Marker: "none",
		Points: histogram(keys, 1000),
	}}
	histPath := "../bench_results/plots/distributions/hist_sosd_osm.svg"
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("Key Density — sosd_osm (n=%d, 1000 bins)", len(keys)),
		XLabel: "Normalized Key Position",
		YLabel: "Relative Density",
		YScale: testutils.Log10,
		XMax:   25,
	}, histSeries, histPath)
	if err != nil {
		t.Errorf("histogram SVG failed: %v", err)
	} else {
		fmt.Printf("Histogram written to %s\n", histPath)
	}

	cdfSeries := []testutils.SeriesData{{
		Name:   "sosd_osm",
		Color:  "#27ae60",
		Marker: "none",
		Points: normalizedCDF(keys, 256),
	}}
	cdfPath := "../bench_results/plots/distributions/cdf_sosd_osm.svg"
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("CDF — sosd_osm (n=%d, normalized)", len(keys)),
		XLabel: "Normalized Key Position",
		YLabel: "Cumulative Fraction",
		XMax:   25,
	}, cdfSeries, cdfPath)
	if err != nil {
		t.Errorf("CDF SVG failed: %v", err)
	} else {
		fmt.Printf("CDF written to %s\n", cdfPath)
	}
}

func TestDistribution_SOSD_Books_Histogram(t *testing.T) {
	path := sosdPath("books_200M_uint32")
	keys, err := loadSOSDUint32(path, 0)
	if err != nil {
		t.Skipf("SOSD books_200M_uint32 not available: %v", err)
	}

	os.MkdirAll("../bench_results/plots/distributions", 0755)

	histSeries := []testutils.SeriesData{{
		Name:   "sosd_books",
		Color:  "#8e44ad",
		Marker: "none",
		Points: histogram(keys, 1000),
	}}
	histPath := "../bench_results/plots/distributions/hist_sosd_books.svg"
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("Key Density — sosd_books (n=%d, 1000 bins)", len(keys)),
		XLabel: "Normalized Key Position",
		YLabel: "Relative Density",
		YScale: testutils.Log10,
		XMax:   25,
	}, histSeries, histPath)
	if err != nil {
		t.Errorf("histogram SVG failed: %v", err)
	} else {
		fmt.Printf("Histogram written to %s\n", histPath)
	}

	cdfSeries := []testutils.SeriesData{{
		Name:   "sosd_books",
		Color:  "#8e44ad",
		Marker: "none",
		Points: normalizedCDF(keys, 256),
	}}
	cdfPath := "../bench_results/plots/distributions/cdf_sosd_books.svg"
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("CDF — sosd_books (n=%d, normalized)", len(keys)),
		XLabel: "Normalized Key Position",
		YLabel: "Cumulative Fraction",
		XMax:   25,
	}, cdfSeries, cdfPath)
	if err != nil {
		t.Errorf("CDF SVG failed: %v", err)
	} else {
		fmt.Printf("CDF written to %s\n", cdfPath)
	}
}
