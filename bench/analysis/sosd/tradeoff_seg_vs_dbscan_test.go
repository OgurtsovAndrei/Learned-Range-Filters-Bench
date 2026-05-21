//go:build heavy

package sosd_test

import (
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"testing"

	"Thesis-bench-industry/bench/datasets"
	"Thesis-bench-industry/bench/internal/querygen"
	"Thesis/emptiness/approx/hybrid/are_dbscan"
	"Thesis/emptiness/approx/hybrid/are_seg"
	"Thesis/testutils"
)

const (
	segN          = 1 << 24
	segRangeLen   = uint64(128)
	segQueryCount = 1 << 15
	segRuns       = 3
)

var segKGrid = []uint32{22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42}

func segDataPath(name string) string {
	_, f, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(f), "..", "..", "sosd_data", name)
}

func segLoadDataset(path string, keyType datasets.SOSDKeyType) ([]uint64, error) {
	r := &datasets.SOSDReader{Path: path, Label: path, KeyType: keyType, MaxKeys: segN}
	return r.Keys()
}

func segMakeQueries(keys []uint64, count int, seed int64) [][2]uint64 {
	rng := rand.New(rand.NewSource(seed))
	return querygen.GenerateSmartQueriesWeighted(keys, count, segRangeLen, querygen.DefaultSmartMix, rng)
}

func segRunTradeoff(t *testing.T, label, svgPath string, keys []uint64) {
	t.Helper()
	n := len(keys)

	seeds := []int64{11111, 22222, 33333}
	querySets := make([][][2]uint64, segRuns)
	for r := 0; r < segRuns; r++ {
		querySets[r] = segMakeQueries(keys, segQueryCount, seeds[r])
		if len(querySets[r]) == 0 {
			t.Fatalf("smart query generator returned 0 queries (run %d)", r)
		}
	}

	type result struct {
		which string
		bpk   float64
		fpr   float64
	}
	results := make([]result, len(segKGrid)*2)

	var wg sync.WaitGroup
	for i, K := range segKGrid {
		for j, which := range []string{"seg", "dbscan"} {
			i, K, j, which := i, K, j, which
			wg.Add(1)
			go func() {
				defer wg.Done()
				var isEmpty func(a, b uint64) bool
				var sizeBits uint64

				switch which {
				case "seg":
					f, err := are_seg.NewSegAREFromK(keys, 64, K, segRangeLen)
					if err != nil {
						return
					}
					isEmpty = f.IsEmpty
					sizeBits = f.SizeInBits()
				case "dbscan":
					f, err := are_dbscan.NewHybridScanARE(keys, 64, are_dbscan.Config{K: K})
					if err != nil {
						return
					}
					isEmpty = f.IsEmpty
					sizeBits = f.SizeInBits()
				}

				bpk := float64(sizeBits) / float64(n)
				var fprSum float64
				for _, qs := range querySets {
					fprSum += testutils.MeasureFPR(keys, qs, isEmpty)
				}
				results[i*2+j] = result{which, bpk, fprSum / float64(segRuns)}
			}()
		}
	}
	wg.Wait()

	segPts := make([]testutils.Point, 0, len(segKGrid))
	dbscanPts := make([]testutils.Point, 0, len(segKGrid))
	for _, r := range results {
		if r.bpk == 0 {
			continue
		}
		pt := testutils.Point{X: r.bpk, Y: r.fpr}
		switch r.which {
		case "seg":
			segPts = append(segPts, pt)
		case "dbscan":
			dbscanPts = append(dbscanPts, pt)
		}
	}

	sortPts := func(pts []testutils.Point) {
		sort.Slice(pts, func(i, j int) bool { return pts[i].X < pts[j].X })
	}
	sortPts(segPts)
	sortPts(dbscanPts)

	// Theoretical: BPK = K for pure fingerprint, FPR = L/2^K
	theoretical := testutils.SeriesData{Name: "Theoretical", Color: "#ef4444", Dashed: true, Marker: "none"}
	for K := uint32(10); K <= 50; K++ {
		eps := float64(segRangeLen) / math.Exp2(float64(K))
		if eps >= 1e-6 && eps <= 1 {
			theoretical.Points = append(theoretical.Points, testutils.Point{X: float64(K), Y: eps})
		}
	}

	yFloor := 1.0 / float64(segQueryCount*segRuns)
	series := []testutils.SeriesData{
		theoretical,
		{Name: "SegARE (minPts=256, no border)", Color: "#e67e22", Marker: "circle", Points: segPts},
		{Name: "Scan-ARE (DBSCAN)", Color: "#3498db", Marker: "diamond", Points: dbscanPts},
	}

	title := fmt.Sprintf("FPR vs BPK — %s  (n=%d, L=%d, smart queries)", label, n, segRangeLen)
	err := testutils.GenerateTradeoffSVG(title, "Bits per Key (BPK)", "False Positive Rate (FPR)",
		series, svgPath, yFloor)
	if err != nil {
		t.Errorf("SVG write failed: %v", err)
	} else {
		t.Logf("SVG → %s", svgPath)
	}
}

func TestSegARE_Tradeoff_SOSD_L128(t *testing.T) {
	dir := filepath.Dir(func() string {
		_, f, _, _ := runtime.Caller(0)
		return f
	}())

	datasets_ := []struct {
		label   string
		path    string
		keyType datasets.SOSDKeyType
	}{
		{"Facebook", segDataPath("fb_200M_uint64"), datasets.SOSDUint64},
		{"Wiki-TS", segDataPath("wiki_ts_200M_uint64"), datasets.SOSDUint64},
		{"OSM", segDataPath("osm_cellids_800M_uint64"), datasets.SOSDUint64},
		{"Books", segDataPath("books_200M_uint32"), datasets.SOSDUint32},
	}

	for _, ds := range datasets_ {
		ds := ds
		t.Run(ds.label, func(t *testing.T) {
			keys, err := segLoadDataset(ds.path, ds.keyType)
			if err != nil {
				t.Skipf("dataset not available: %v", err)
			}
			t.Logf("Loaded %d keys from %s", len(keys), ds.label)

			svgName := fmt.Sprintf("tradeoff_seg_vs_dbscan_%s_L128.svg", ds.label)
			svgPath := filepath.Join(dir, svgName)
			segRunTradeoff(t, ds.label, svgPath, keys)
		})
	}
}
