package bench_test

import (
	"Thesis/testutils"
	"encoding/binary"
	"fmt"
	"os"
	"sort"
	"testing"
)

// loadSOSDUint32 reads a SOSD binary file with uint32 keys:
// [uint64 count][count × uint32 keys], returning them as []uint64.
// If maxKeys <= 0, all keys are returned.
func loadSOSDUint32(path string, maxKeys int) ([]uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var count uint64
	if err := binary.Read(f, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("read count: %w", err)
	}

	readN := int(count)
	if maxKeys > 0 && maxKeys < readN {
		readN = maxKeys
	}

	raw := make([]uint32, readN)
	if err := binary.Read(f, binary.LittleEndian, raw); err != nil {
		return nil, fmt.Errorf("read keys: %w", err)
	}

	keys := make([]uint64, readN)
	for i, v := range raw {
		keys[i] = uint64(v)
	}

	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	j := 0
	for i := 1; i < len(keys); i++ {
		if keys[i] != keys[j] {
			j++
			keys[j] = keys[i]
		}
	}
	return keys[:j+1], nil
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
	}, cdfSeries, cdfPath)
	if err != nil {
		t.Errorf("CDF SVG failed: %v", err)
	} else {
		fmt.Printf("CDF written to %s\n", cdfPath)
	}
}
