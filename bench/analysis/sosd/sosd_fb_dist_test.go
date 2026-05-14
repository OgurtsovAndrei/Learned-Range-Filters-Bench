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

func loadFacebookKeys(n int) ([]uint64, error) {
	return keygen.LoadSOSDUint64(keygen.SOSDPath("fb_200M_uint64"), n)
}

func loadWikiKeys(n int) ([]uint64, error) {
	return keygen.LoadSOSDUint64(keygen.SOSDPath("wiki_ts_200M_uint64"), n)
}

func loadOSMKeys(n int) ([]uint64, error) {
	return keygen.LoadSOSDUint64(keygen.SOSDPath("osm_cellids_800M_uint64"), n)
}

func loadBooksKeys(n int) ([]uint64, error) {
	return keygen.LoadSOSDUint32(keygen.SOSDPath("books_200M_uint32"), n)
}

func histogram(keys []uint64, nBins int) []testutils.Point {
	return benchutil.Histogram(keys, nBins)
}

func normalizedCDF(keys []uint64, sampleEvery int) []testutils.Point {
	return benchutil.NormalizedCDF(keys, sampleEvery)
}



func TestDistribution_SOSD_FB_Histogram(t *testing.T) {

	keys, err := loadFacebookKeys(0)
	if err != nil {
		t.Skipf("SOSD fb_200M_uint64 not available: %v", err)
	}

	os.MkdirAll("../../../Thesis/text/plots/distributions", 0755)

	// Histogram
	histSeries := []testutils.SeriesData{{
		Name:   "Facebook",
		Color:  "#e74c3c",
		Marker: "none",
		Points: histogram(keys, 1000),
	}}
	histPath := "../../../Thesis/text/plots/distributions/hist_sosd_fb.svg"
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:         "Facebook",
		XLabel:        "Normalized Key Position",
		YLabel:        "Relative Density",
		YScale:        testutils.Log10,
		KeepAllPoints: true,
		YCeil:         1.0,
		XMax:          25,
	}, histSeries, histPath)
	if err != nil {
		t.Errorf("histogram SVG failed: %v", err)
	} else {
		fmt.Printf("Histogram written to %s\n", histPath)
	}

	// CDF
	cdfSeries := []testutils.SeriesData{{
		Name:   "Facebook",
		Color:  "#e74c3c",
		Marker: "none",
		Points: normalizedCDF(keys, 256),
	}}
	cdfPath := "../../../Thesis/text/plots/distributions/cdf_sosd_fb.svg"
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("Facebook CDF (n=%d, normalized)", len(keys)),
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
	keys, err := loadWikiKeys(0)
	if err != nil {
		t.Skipf("SOSD wiki_ts_200M_uint64 not available: %v", err)
	}

	os.MkdirAll("../../../Thesis/text/plots/distributions", 0755)

	histSeries := []testutils.SeriesData{{
		Name:   "Wiki",
		Color:  "#3498db",
		Marker: "none",
		Points: histogram(keys, 1000),
	}}
	histPath := "../../../Thesis/text/plots/distributions/hist_sosd_wiki.svg"
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:         "Wiki",
		XLabel:        "Normalized Key Position",
		YLabel:        "Relative Density",
		YScale:        testutils.Log10,
		KeepAllPoints: true,
		XMax:          25,
	}, histSeries, histPath)
	if err != nil {
		t.Errorf("histogram SVG failed: %v", err)
	} else {
		fmt.Printf("Histogram written to %s\n", histPath)
	}

	cdfSeries := []testutils.SeriesData{{
		Name:   "Wiki",
		Color:  "#3498db",
		Marker: "none",
		Points: normalizedCDF(keys, 256),
	}}
	cdfPath := "../../../Thesis/text/plots/distributions/cdf_sosd_wiki.svg"
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("Wiki CDF (n=%d, normalized)", len(keys)),
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
	keys, err := loadOSMKeys(0)
	if err != nil {
		t.Skipf("SOSD osm_cellids_800M_uint64 not available: %v", err)
	}

	os.MkdirAll("../../../Thesis/text/plots/distributions", 0755)

	histSeries := []testutils.SeriesData{{
		Name:   "OSM",
		Color:  "#27ae60",
		Marker: "none",
		Points: histogram(keys, 1000),
	}}
	histPath := "../../../Thesis/text/plots/distributions/hist_sosd_osm.svg"
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:         "OSM",
		XLabel:        "Normalized Key Position",
		YLabel:        "Relative Density",
		YScale:        testutils.Log10,
		KeepAllPoints: true,
		XMax:          25,
	}, histSeries, histPath)
	if err != nil {
		t.Errorf("histogram SVG failed: %v", err)
	} else {
		fmt.Printf("Histogram written to %s\n", histPath)
	}

	cdfSeries := []testutils.SeriesData{{
		Name:   "OSM",
		Color:  "#27ae60",
		Marker: "none",
		Points: normalizedCDF(keys, 256),
	}}
	cdfPath := "../../../Thesis/text/plots/distributions/cdf_sosd_osm.svg"
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("OSM CDF (n=%d, normalized)", len(keys)),
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
	keys, err := loadBooksKeys(0)
	if err != nil {
		t.Skipf("SOSD books_200M_uint32 not available: %v", err)
	}

	os.MkdirAll("../../../Thesis/text/plots/distributions", 0755)

	histSeries := []testutils.SeriesData{{
		Name:   "Books",
		Color:  "#8e44ad",
		Marker: "none",
		Points: histogram(keys, 1000),
	}}
	histPath := "../../../Thesis/text/plots/distributions/hist_sosd_books.svg"
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:         "Books",
		XLabel:        "Normalized Key Position",
		YLabel:        "Relative Density",
		YScale:        testutils.Log10,
		KeepAllPoints: true,
		XMax:          25,
	}, histSeries, histPath)
	if err != nil {
		t.Errorf("histogram SVG failed: %v", err)
	} else {
		fmt.Printf("Histogram written to %s\n", histPath)
	}

	cdfSeries := []testutils.SeriesData{{
		Name:   "Books",
		Color:  "#8e44ad",
		Marker: "none",
		Points: normalizedCDF(keys, 256),
	}}
	cdfPath := "../../../Thesis/text/plots/distributions/cdf_sosd_books.svg"
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("Books CDF (n=%d, normalized)", len(keys)),
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
