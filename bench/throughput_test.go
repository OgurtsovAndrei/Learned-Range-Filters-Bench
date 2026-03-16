package bench_test

import (
	"Thesis-bench-industry/snarf"
	"Thesis-bench-industry/surf"
	"Thesis/bits"
	"Thesis/emptiness/are_bloom"
	"Thesis/emptiness/are_hybrid"
	"Thesis/emptiness/are_optimized"
	"Thesis/emptiness/are_pgm"
	"Thesis/emptiness/are_soda_hash"
	"Thesis/emptiness/are_trunc"
	"Thesis/testutils"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"
)

func syntheticDataPath(name string) string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), "synthetic_data", name)
}

func writeSOSDUint64(path string, keys []uint64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	count := uint64(len(keys))
	binary.Write(f, binary.LittleEndian, count)
	return binary.Write(f, binary.LittleEndian, keys)
}

func TestGenerateSyntheticKeys(t *testing.T) {
	dir := syntheticDataPath("")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatalf("mkdir synthetic_data: %v", err)
	}

	regen := os.Getenv("REGEN") != ""

	type distSpec struct {
		file string
		gen  func() []uint64
	}

	specs := []distSpec{
		{
			"clustered_16M_uint64",
			func() []uint64 {
				rng := rand.New(rand.NewSource(99))
				raw, _ := testutils.GenerateClusterDistribution(1<<24, 5, 0.15, rng)
				return mask60Keys(raw)
			},
		},
		{
			"uniform_16M_uint64",
			func() []uint64 {
				rng := rand.New(rand.NewSource(42))
				return generateUniformKeys(1<<24, rng)
			},
		},
		{
			"spread_16M_uint64",
			func() []uint64 {
				return generateSpreadKeys(1 << 24)
			},
		},
		{
			"zipfian_16M_uint64",
			func() []uint64 {
				rng := rand.New(rand.NewSource(77))
				keys, _ := generateZipfianKeys(1<<24, 1000, rng)
				return keys
			},
		},
	}

	type result struct {
		file string
		keys []uint64
	}
	results := make([]result, len(specs))

	var wg sync.WaitGroup
	for i, spec := range specs {
		path := syntheticDataPath(spec.file)
		if !regen {
			if _, err := os.Stat(path); err == nil {
				t.Logf("skipping %s (already exists; set REGEN=1 to regenerate)", spec.file)
				continue
			}
		}
		i, spec := i, spec
		wg.Add(1)
		go func() {
			defer wg.Done()
			t.Logf("generating %s ...", spec.file)
			results[i] = result{file: spec.file, keys: spec.gen()}
		}()
	}
	wg.Wait()

	for _, r := range results {
		if r.keys == nil {
			continue
		}
		path := syntheticDataPath(r.file)
		if err := writeSOSDUint64(path, r.keys); err != nil {
			t.Errorf("write %s: %v", r.file, err)
		} else {
			t.Logf("wrote %s (%d keys)", r.file, len(r.keys))
		}
	}
}

func TestBuildThroughput(t *testing.T) {
	nSizes := []int{1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24}

	const (
		eps      = 0.01
		rangeLen = uint64(100)
	)

	bpk := math.Log2(float64(rangeLen) / eps)

	type buildFilterDef struct {
		name   string
		color  string
		marker string
		dashed bool
		build  func(keys []uint64) error
	}

	filters := []buildFilterDef{
		{
			name: "Adaptive(t=0)", color: "#2a7fff", marker: "square", dashed: false,
			build: func(keys []uint64) error {
				bs := make([]bits.BitString, len(keys))
				for i, v := range keys {
					bs[i] = testutils.TrieBS(v)
				}
				_, err := are_optimized.NewOptimizedARE(bs, rangeLen, eps, 0)
				return err
			},
		},
		{
			name: "SODA", color: "#4dd88a", marker: "diamond", dashed: false,
			build: func(keys []uint64) error {
				_, err := are_soda_hash.NewApproximateRangeEmptinessSoda(keys, rangeLen, eps)
				return err
			},
		},
		{
			name: "Truncation", color: "#9b59b6", marker: "triangle", dashed: false,
			build: func(keys []uint64) error {
				bs := make([]bits.BitString, len(keys))
				for i, v := range keys {
					bs[i] = testutils.TrieBS(v)
				}
				_, err := are_trunc.NewApproximateRangeEmptiness(bs, eps)
				return err
			},
		},
		{
			name: "Hybrid", color: "#ff6b6b", marker: "star", dashed: false,
			build: func(keys []uint64) error {
				bs := make([]bits.BitString, len(keys))
				for i, v := range keys {
					bs[i] = testutils.TrieBS(v)
				}
				_, err := are_hybrid.NewHybridARE(bs, rangeLen, eps)
				return err
			},
		},
		{
			name: "CDF-ARE", color: "#ff922b", marker: "circle", dashed: false,
			build: func(keys []uint64) error {
				_, err := are_pgm.NewPGMApproximateRangeEmptiness(keys, rangeLen, eps, 64)
				return err
			},
		},
		{
			name: "BloomARE", color: "#888888", marker: "circle", dashed: true,
			build: func(keys []uint64) error {
				_, err := are_bloom.NewBloomARE(keys, rangeLen, eps)
				return err
			},
		},
		{
			name: "Grafite", color: "#1a6b3c", marker: "diamond", dashed: false,
			build: func(keys []uint64) error {
				if tryGrafite(keys, bpk) == nil {
					return fmt.Errorf("grafite: unsupported bpk=%.2f for this key set", bpk)
				}
				return nil
			},
		},
		{
			name: "SNARF", color: "#1a3a6b", marker: "star", dashed: false,
			build: func(keys []uint64) error {
				snarf.New(keys, bpk)
				return nil
			},
		},
		{
			name: "SuRF", color: "#111111", marker: "square", dashed: false,
			build: func(keys []uint64) error {
				surf.New(keys, surf.SuffixNone, 0, 0)
				return nil
			},
		},
		{
			name: "SuRFHash(8)", color: "#111111", marker: "triangle", dashed: false,
			build: func(keys []uint64) error {
				surf.New(keys, surf.SuffixHash, 8, 0)
				return nil
			},
		},
		{
			name: "SuRFReal(8)", color: "#111111", marker: "diamond", dashed: true,
			build: func(keys []uint64) error {
				surf.New(keys, surf.SuffixReal, 0, 8)
				return nil
			},
		},
	}

	distDefs := []struct{ name, file string }{
		{"clustered", "clustered_16M_uint64"},
		{"uniform", "uniform_16M_uint64"},
		{"spread", "spread_16M_uint64"},
		{"zipfian", "zipfian_16M_uint64"},
	}

	for _, dist := range distDefs {
		dist := dist
		t.Run(dist.name, func(t *testing.T) {
			allKeys, err := loadSOSDUint64(syntheticDataPath(dist.file), 0)
			if err != nil {
				t.Skipf("run TestGenerateSyntheticKeys first: %v", err)
			}

			allSeries := map[string]*testutils.SeriesData{
				"Adaptive(t=0)": {Name: "Adaptive(t=0)", Color: "#2a7fff", Marker: "square"},
				"SODA":          {Name: "SODA", Color: "#4dd88a", Marker: "diamond"},
				"Truncation":    {Name: "Truncation", Color: "#9b59b6", Marker: "triangle"},
				"Hybrid":        {Name: "Hybrid", Color: "#ff6b6b", Marker: "star"},
				"CDF-ARE":       {Name: "CDF-ARE", Color: "#ff922b", Marker: "circle"},
				"BloomARE":      {Name: "BloomARE", Color: "#888888", Marker: "circle", Dashed: true},
				"Grafite":       {Name: "Grafite", Color: "#1a6b3c", Marker: "diamond"},
				"SNARF":         {Name: "SNARF", Color: "#1a3a6b", Marker: "star"},
				"SuRF":          {Name: "SuRF", Color: "#111111", Marker: "square"},
				"SuRFHash(8)":   {Name: "SuRFHash(8)", Color: "#111111", Marker: "triangle"},
				"SuRFReal(8)":   {Name: "SuRFReal(8)", Color: "#111111", Marker: "diamond", Dashed: true},
			}

			dataDir := "../bench_results/data/build_throughput"
			os.MkdirAll(dataDir, 0755)
			dataPath := fmt.Sprintf("%s/%s.json", dataDir, dist.name)

			if os.Getenv("PLOT_ONLY") != "" {
				if err := loadSeriesData(dataPath, allSeries); err != nil {
					t.Skipf("no saved data for %s: %v", dist.name, err)
					return
				}
				fmt.Printf("\n=== Plot-only mode — build throughput %s (loaded from %s) ===\n", dist.name, dataPath)
			} else {
				fmt.Printf("\n=== Build Throughput — %s (ε=%.2f, L=%d) ===\n", dist.name, eps, rangeLen)
				fmt.Printf("%-16s | %-8s | %-16s | %-10s\n", "Filter", "N", "Throughput", "Time")

				for _, fd := range filters {
					// warm-up
					fd.build(allKeys[:1<<10])
					runtime.GC()

					for _, n := range nSizes {
						keys := allKeys[:n]
						start := time.Now()
						err := fd.build(keys)
						dur := time.Since(start)
						if err != nil {
							continue
						}
						mPerSec := float64(n) / dur.Seconds() / 1e6
						allSeries[fd.name].Points = append(allSeries[fd.name].Points,
							testutils.Point{X: float64(n), Y: mPerSec})
						fmt.Printf("%-16s | N=%-8d | %.2f M keys/sec | %.1f ms\n",
							fd.name, n, mPerSec, dur.Seconds()*1000)
					}
				}

				if err := saveSeriesData(dataPath, allSeries); err != nil {
					t.Logf("warning: failed to save data: %v", err)
				}
			}

			os.MkdirAll("../bench_results/plots/build_throughput", 0755)
			svgPath := fmt.Sprintf("../bench_results/plots/build_throughput/%s.svg", dist.name)

			orderedSeries := []testutils.SeriesData{
				*allSeries["Adaptive(t=0)"],
				*allSeries["SODA"],
				*allSeries["Truncation"],
				*allSeries["Hybrid"],
				*allSeries["CDF-ARE"],
				*allSeries["BloomARE"],
				*allSeries["Grafite"],
				*allSeries["SNARF"],
				*allSeries["SuRF"],
				*allSeries["SuRFHash(8)"],
				*allSeries["SuRFReal(8)"],
			}

			if err := testutils.GeneratePerformanceSVG(testutils.PlotConfig{
				Title:  fmt.Sprintf("Build Throughput — %s (ε=0.01, L=100)", dist.name),
				XLabel: "Number of Keys (N)",
				YLabel: "Build Throughput (M keys/sec)",
				XScale: testutils.Log10,
				YScale: testutils.Log10,
			}, orderedSeries, svgPath); err != nil {
				t.Errorf("SVG generation failed: %v", err)
			} else {
				fmt.Printf("\nSVG written to %s\n", svgPath)
			}
		})
	}
}

