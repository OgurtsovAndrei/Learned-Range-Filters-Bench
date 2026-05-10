package datasets_test

import (
	"fmt"
	"math/bits"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"Thesis-bench-industry/bench/datasets"
	"Thesis/testutils"
)

// TestSOSD_Books800M_Distribution loads sosd books_800M_uint64, prints gap-
// percentile statistics, identifies cluster boundaries, and writes 1000-bin
// histograms in log + linear + smoothed (window 5/7/11) variants.
func TestSOSD_Books800M_Distribution(t *testing.T) {
	r := &datasets.SOSDReader{
		Path:    sosdPath("books_800M_uint64"),
		Label:   "sosd_books_800M",
		KeyType: datasets.SOSDUint64,
	}
	keys, err := r.Keys()
	if err != nil {
		t.Skipf("books_800M_uint64 not available: %v", err)
	}
	n := len(keys)
	if n < 2 {
		t.Fatalf("not enough keys: %d", n)
	}

	span := keys[n-1] - keys[0]
	meanGap := float64(span) / float64(n-1)
	fmt.Printf("\n=== sosd_books_800M (uint64) ===\n")
	fmt.Printf("n     = %d\n", n)
	fmt.Printf("range = [%d, %d]\n", keys[0], keys[n-1])
	fmt.Printf("span  = %d  (~2^%.1f)\n", span, log2u64(span))
	fmt.Printf("mean gap = %.0f  (~2^%.1f)\n", meanGap, log2u64(uint64(meanGap)))

	gaps := make([]uint64, n-1)
	for i := 1; i < n; i++ {
		gaps[i-1] = keys[i] - keys[i-1]
	}
	sortedGaps := make([]uint64, len(gaps))
	copy(sortedGaps, gaps)
	sort.Slice(sortedGaps, func(i, j int) bool { return sortedGaps[i] < sortedGaps[j] })
	pct := func(p float64) uint64 {
		idx := int(p * float64(len(sortedGaps)-1))
		return sortedGaps[idx]
	}

	fmt.Printf("\ngap percentiles:\n")
	for _, p := range []float64{0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999} {
		g := pct(p)
		fmt.Printf("  p%-7s = %20d  (%.2fx mean)\n",
			fmt7gPct(p), g, float64(g)/meanGap)
	}
	fmt.Printf("  max      = %20d  (%.2fx mean)\n",
		sortedGaps[len(sortedGaps)-1],
		float64(sortedGaps[len(sortedGaps)-1])/meanGap)

	fmt.Printf("\ncluster boundaries:\n")
	for _, k := range []float64{10, 100, 1000, 10000, 100000, 1000000} {
		thr := uint64(k * meanGap)
		cnt := 0
		for _, g := range gaps {
			if g >= thr {
				cnt++
			}
		}
		fmt.Printf("  gaps ≥ %7.0fx mean: %d  (=> %d clusters)\n",
			k, cnt, cnt+1)
	}

	histDir := filepath.Join(filepath.Dir(r.Path), "..", "bench_results",
		"plots", "distributions")
	histDirLinear := filepath.Join(filepath.Dir(r.Path), "..", "bench_results",
		"plots", "distributions_linear")
	histDirSmooth := filepath.Join(filepath.Dir(r.Path), "..", "bench_results",
		"plots", "distributions_smoothed")
	os.MkdirAll(histDir, 0755)
	os.MkdirAll(histDirLinear, 0755)
	os.MkdirAll(histDirSmooth, 0755)

	const nBins = 1000
	minK, maxK := keys[0], keys[n-1]
	binSpan := maxK - minK
	if binSpan == 0 {
		binSpan = 1
	}
	binWidth := binSpan / nBins
	if binWidth == 0 {
		binWidth = 1
	}
	counts := make([]uint64, nBins)
	for _, k := range keys {
		b := int((k - minK) / binWidth)
		if b >= nBins {
			b = nBins - 1
		}
		counts[b]++
	}
	var maxCount uint64
	for _, c := range counts {
		if c > maxCount {
			maxCount = c
		}
	}

	ptsLog := make([]testutils.Point, nBins)
	ptsLin := make([]testutils.Point, nBins)
	for i, c := range counts {
		x := float64(i) / float64(nBins)
		yLin := float64(c) / float64(maxCount)
		yLog := yLin
		if yLog == 0 {
			yLog = 1e-9
		}
		ptsLog[i] = testutils.Point{X: x, Y: yLog}
		ptsLin[i] = testutils.Point{X: x, Y: yLin}
	}
	title := fmt.Sprintf("Key Density — sosd_books_800M (n=%d, %d bins)", n, nBins)

	logPath := filepath.Join(histDir, "hist_sosd_books_800M.svg")
	if err := testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title: title, XLabel: "Normalized Key Position",
		YLabel: "Relative Density",
		YScale: testutils.Log10, XMax: 25,
	}, []testutils.SeriesData{{
		Name: r.Label, Color: "#673ab7", Marker: "none", Points: ptsLog,
	}}, logPath); err != nil {
		t.Fatalf("log svg: %v", err)
	}
	fmt.Printf("\nsvg → %s\n", logPath)

	linPath := filepath.Join(histDirLinear, "hist_sosd_books_800M.svg")
	if err := testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title: title, XLabel: "Normalized Key Position",
		YLabel: "Density (bin / max)",
		YScale: testutils.Linear, XMax: 25,
	}, []testutils.SeriesData{{
		Name: r.Label, Color: "#673ab7", Marker: "none", Points: ptsLin,
	}}, linPath); err != nil {
		t.Fatalf("linear svg: %v", err)
	}
	fmt.Printf("svg → %s\n", linPath)

	for _, w := range []int{5, 7, 11} {
		smoothed := movingAverage(ptsLin, w)
		path := filepath.Join(histDirSmooth,
			fmt.Sprintf("hist_sosd_books_800M_w%d.svg", w))
		if err := testutils.GeneratePerformanceSVG(testutils.PlotConfig{
			Title:  fmt.Sprintf("%s — smoothed (window=%d bins)", title, w),
			XLabel: "Normalized Key Position",
			YLabel: fmt.Sprintf("Mean of %d-bin window", w),
			YScale: testutils.Linear, XMax: 25,
		}, []testutils.SeriesData{{
			Name: r.Label, Color: "#673ab7", Marker: "none", Points: smoothed,
		}}, path); err != nil {
			t.Fatalf("smoothed svg w=%d: %v", w, err)
		}
		fmt.Printf("svg → %s\n", path)
	}

	// Dense-head zoom: cut at first super-gap (≥ 10x mean), re-bin remainder.
	cutAt := n
	threshold := uint64(10 * meanGap)
	for i := 1; i < n; i++ {
		if keys[i]-keys[i-1] >= threshold {
			cutAt = i
			break
		}
	}
	if cutAt < n {
		dense := keys[:cutAt]
		dn := len(dense)
		dminK, dmaxK := dense[0], dense[dn-1]
		dspan := dmaxK - dminK
		if dspan == 0 {
			dspan = 1
		}
		dBinWidth := dspan / nBins
		if dBinWidth == 0 {
			dBinWidth = 1
		}
		dCounts := make([]uint64, nBins)
		for _, k := range dense {
			b := int((k - dminK) / dBinWidth)
			if b >= nBins {
				b = nBins - 1
			}
			dCounts[b]++
		}
		var dMaxCount uint64
		for _, c := range dCounts {
			if c > dMaxCount {
				dMaxCount = c
			}
		}
		fmt.Printf("\n--- dense head (cut at first gap >= %dx mean) ---\n", 10)
		fmt.Printf("dense keys: %d (%.4f%% of total)\n", dn, 100*float64(dn)/float64(n))
		fmt.Printf("dense span: %d (~2^%.1f, vs full 2^%.1f)\n",
			dspan, log2u64(uint64(dspan)), log2u64(span))

		dPts := make([]testutils.Point, nBins)
		for i, c := range dCounts {
			y := float64(c) / float64(dMaxCount)
			if y == 0 {
				y = 1e-9
			}
			dPts[i] = testutils.Point{X: float64(i) / float64(nBins), Y: y}
		}
		dPath := filepath.Join(histDir, "hist_sosd_books_800M_densehead.svg")
		if err := testutils.GeneratePerformanceSVG(testutils.PlotConfig{
			Title: fmt.Sprintf("Key Density — sosd_books_800M dense head (n=%d, 1000 bins)", dn),
			XLabel: "Normalized Position (within dense head)",
			YLabel: "Relative Density",
			YScale: testutils.Log10, XMax: 25,
		}, []testutils.SeriesData{{
			Name: r.Label, Color: "#673ab7", Marker: "none", Points: dPts,
		}}, dPath); err != nil {
			t.Fatalf("dense head svg: %v", err)
		}
		fmt.Printf("svg → %s\n", dPath)
	}
}

func fmt7gPct(p float64) string {
	switch p {
	case 0.5:
		return "50"
	case 0.9:
		return "90"
	case 0.99:
		return "99"
	case 0.999:
		return "99.9"
	case 0.9999:
		return "99.99"
	case 0.99999:
		return "99.999"
	}
	return fmt.Sprintf("%.5g", p*100)
}

func log2u64(x uint64) float64 {
	if x == 0 {
		return 0
	}
	return float64(63 - bits.LeadingZeros64(x))
}
