package bench_test

import (
	"Thesis-bench-industry/grafite"
	"Thesis-bench-industry/snarf"
	"Thesis-bench-industry/surf"
	"Thesis/bits"
	"Thesis/emptiness/are"
	"Thesis/emptiness/are_bloom"
	"Thesis/emptiness/are_hybrid"
	"Thesis/emptiness/are_optimized"
	"Thesis/emptiness/are_pgm"
	"Thesis/emptiness/are_soda_hash"
	"Thesis/testutils"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"testing"
)

const mask60 = (uint64(1) << 60) - 1

func mask60Keys(raw []uint64) []uint64 {
	seen := make(map[uint64]bool, len(raw))
	masked := make([]uint64, 0, len(raw))
	for _, k := range raw {
		k &= mask60
		if !seen[k] {
			seen[k] = true
			masked = append(masked, k)
		}
	}
	sort.Slice(masked, func(i, j int) bool { return masked[i] < masked[j] })
	return masked
}

func mask60Queries(queries [][2]uint64) [][2]uint64 {
	out := make([][2]uint64, len(queries))
	for i, q := range queries {
		a := q[0] & mask60
		b := q[1] & mask60
		if b < a {
			// rangeLen crossed the mask boundary; just use a single-point query
			b = a
		}
		out[i] = [2]uint64{a, b}
	}
	return out
}

// benchConfig parameterises a single distribution benchmark.
type benchConfig struct {
	distName  string
	keys      []uint64
	queryFunc func(rangeLen uint64, seed int64) [][2]uint64
}

// tryGrafite returns nil when the requested bpk exceeds what the key universe can support.
func tryGrafite(keys []uint64, bpk float64) *grafite.GrafiteFilter {
	if len(keys) < 2 {
		return nil
	}
	universe := keys[len(keys)-1] - keys[0]
	if universe == 0 {
		return nil
	}
	maxBPK := math.Log2(float64(universe)/float64(len(keys))) + 2
	if bpk > maxBPK {
		return nil
	}
	return grafite.New(keys, bpk)
}

func avgFPR(keys []uint64, querySets [][][2]uint64, isEmpty func(a, b uint64) bool) float64 {
	results := make([]float64, len(querySets))
	var wg sync.WaitGroup
	for i, qs := range querySets {
		wg.Add(1)
		go func(idx int, q [][2]uint64) {
			defer wg.Done()
			results[idx] = testutils.MeasureFPR(keys, q, isEmpty)
		}(i, qs)
	}
	wg.Wait()
	sum := 0.0
	for _, v := range results {
		sum += v
	}
	return sum / float64(len(results))
}

func avgFPRSeq(keys []uint64, querySets [][][2]uint64, isEmpty func(a, b uint64) bool) float64 {
	sum := 0.0
	for _, qs := range querySets {
		sum += testutils.MeasureFPR(keys, qs, isEmpty)
	}
	return sum / float64(len(querySets))
}

type seriesPoint struct {
	series string
	point  testutils.Point
	label  string
}

func runTradeoffBench(t *testing.T, cfg benchConfig) {
	const nRuns = 3

	rangeLens := []uint64{1, 16, 128, 1024}
	bpkSweep := []float64{4, 6, 8, 10, 12, 14, 16, 18, 20}
	epsilons := []float64{0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001}

	keysBS := make([]bits.BitString, len(cfg.keys))
	for i, v := range cfg.keys {
		keysBS[i] = testutils.TrieBS(v)
	}

	os.MkdirAll(fmt.Sprintf("../bench_results/plots/%s", cfg.distName), 0755)

	for _, rangeLen := range rangeLens {
		t.Run(fmt.Sprintf("L=%d", rangeLen), func(t *testing.T) {
			seeds := []int64{12345, 54321, 99999}
			querySets := make([][][2]uint64, nRuns)
			for r := 0; r < nRuns; r++ {
				querySets[r] = cfg.queryFunc(rangeLen, seeds[r])
			}

			// ---- series map ----
			allSeries := map[string]*testutils.SeriesData{
				"Theoretical":    {Name: "Theoretical", Color: "#ef4444", Dashed: true, Marker: "circle"},
				"Grafite":        {Name: "Grafite", Color: "#22a06b", Marker: "diamond"},
				"SNARF":          {Name: "SNARF", Color: "#9b59b6", Marker: "star"},
				"SuRF":           {Name: "SuRF", Color: "#1a9cdb", Marker: "square"},
				"SuRFHash(8)":    {Name: "SuRFHash(8)", Color: "#0e6ea8", Marker: "square"},
				"SuRFReal(8)":    {Name: "SuRFReal(8)", Color: "#084d76", Marker: "square"},
				"Adaptive (t=0)": {Name: "Adaptive (t=0)", Color: "#2a7fff", Marker: "square"},
				"SODA":           {Name: "SODA", Color: "#3aa06b", Marker: "diamond"},
				"Hybrid":         {Name: "Hybrid", Color: "#c0392b", Marker: "star"},
				"CDF-ARE":        {Name: "CDF-ARE", Color: "#e05d10", Marker: "circle"},
				"BloomARE":       {Name: "BloomARE", Color: "#888888", Dashed: true, Marker: "circle"},
			}

			fmt.Printf("\n=== Industry Comparison — %s (60-bit keys, %d keys, L=%d, %d runs) ===\n", cfg.distName, len(cfg.keys), rangeLen, nRuns)
			fmt.Printf("%-16s | %8s | %14s\n", "Series", "BPK", "FPR(avg)")
			fmt.Println(strings.Repeat("-", 45))

			// ---- Theoretical (no measurement needed) ----
			for _, eps := range epsilons {
				thBPK := math.Log2(float64(rangeLen) / eps)
				allSeries["Theoretical"].Points = append(allSeries["Theoretical"].Points,
					testutils.Point{X: thBPK, Y: eps})
			}

			// ---- Build & measure ARE filters in parallel (pure Go, thread-safe) ----
			type fprTask struct {
				series  string
				label   string
				bpk     float64
				isEmpty func(a, b uint64) bool
			}
			var goTasks []fprTask

			for _, eps := range epsilons {
				if f, err := are_optimized.NewOptimizedARE(keysBS, rangeLen, eps, 0); err == nil {
					bpk := float64(f.SizeInBits()) / float64(len(cfg.keys))
					goTasks = append(goTasks, fprTask{"Adaptive (t=0)", "Adaptive(t=0)", bpk,
						func(a, b uint64) bool { return f.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) }})
				}
				if f, err := are_soda_hash.NewApproximateRangeEmptinessSoda(cfg.keys, rangeLen, eps); err == nil {
					bpk := float64(f.SizeInBits()) / float64(len(cfg.keys))
					goTasks = append(goTasks, fprTask{"SODA", "SODA", bpk,
						func(a, b uint64) bool { return f.IsEmpty(a, b) }})
				}
				if f, err := are_hybrid.NewHybridARE(keysBS, rangeLen, eps); err == nil {
					bpk := float64(f.SizeInBits()) / float64(len(cfg.keys))
					goTasks = append(goTasks, fprTask{"Hybrid", "Hybrid", bpk,
						func(a, b uint64) bool { return f.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) }})
				}
				if f, err := are_pgm.NewPGMApproximateRangeEmptiness(cfg.keys, rangeLen, eps, 64); err == nil {
					bpk := float64(f.TotalSizeInBits()) / float64(len(cfg.keys))
					goTasks = append(goTasks, fprTask{"CDF-ARE", "CDF-ARE", bpk,
						func(a, b uint64) bool { return f.IsEmpty(a, b) }})
				}
				if f, err := are_bloom.NewBloomARE(cfg.keys, rangeLen, eps); err == nil {
					bpk := float64(f.SizeInBits()) / float64(len(cfg.keys))
					goTasks = append(goTasks, fprTask{"BloomARE", "BloomARE", bpk,
						func(a, b uint64) bool { return f.IsEmpty(a, b) }})
				}
			}

			goResults := make([]seriesPoint, len(goTasks))
			var wg sync.WaitGroup
			for i, task := range goTasks {
				i, task := i, task
				wg.Add(1)
				go func() {
					defer wg.Done()
					fpr := avgFPR(cfg.keys, querySets, task.isEmpty)
					goResults[i] = seriesPoint{task.series, testutils.Point{X: task.bpk, Y: fpr}, task.label}
				}()
			}
			wg.Wait()

			for _, sp := range goResults {
				allSeries[sp.series].Points = append(allSeries[sp.series].Points, sp.point)
				fmt.Printf("%-16s | %8.2f | %14.6f\n", sp.label, sp.point.X, sp.point.Y)
			}

			// ---- CGo filters: build & measure sequentially (not thread-safe) ----
			for _, bpk := range bpkSweep {
				if f := tryGrafite(cfg.keys, bpk); f != nil {
					actualBPK := float64(f.SizeInBits()) / float64(len(cfg.keys))
					fpr := avgFPRSeq(cfg.keys, querySets, func(a, b uint64) bool { return f.IsEmpty(a, b) })
					allSeries["Grafite"].Points = append(allSeries["Grafite"].Points,
						testutils.Point{X: actualBPK, Y: fpr})
					fmt.Printf("%-16s | %8.2f | %14.6f\n", fmt.Sprintf("Grafite(bpk=%.0f)", bpk), actualBPK, fpr)
				}

				f := snarf.New(cfg.keys, bpk)
				actualBPK := float64(f.SizeInBits()) / float64(len(cfg.keys))
				fpr := avgFPRSeq(cfg.keys, querySets, func(a, b uint64) bool { return f.IsEmpty(a, b) })
				allSeries["SNARF"].Points = append(allSeries["SNARF"].Points,
					testutils.Point{X: actualBPK, Y: fpr})
				fmt.Printf("%-16s | %8.2f | %14.6f\n", fmt.Sprintf("SNARF(bpk=%.0f)", bpk), actualBPK, fpr)
			}

			type surfVariant struct {
				name     string
				st       surf.SuffixType
				hashBits int
				realBits int
			}
			for _, sv := range []surfVariant{
				{"SuRF", surf.SuffixNone, 0, 0},
				{"SuRFHash(8)", surf.SuffixHash, 8, 0},
				{"SuRFReal(8)", surf.SuffixReal, 0, 8},
			} {
				f := surf.New(cfg.keys, sv.st, sv.hashBits, sv.realBits)
				actualBPK := float64(f.SizeInBits()) / float64(len(cfg.keys))
				fpr := avgFPRSeq(cfg.keys, querySets, func(a, b uint64) bool { return f.IsEmpty(a, b) })
				allSeries[sv.name].Points = append(allSeries[sv.name].Points,
					testutils.Point{X: actualBPK, Y: fpr})
				fmt.Printf("%-16s | %8.2f | %14.6f\n", sv.name, actualBPK, fpr)
			}

			// ---- Generate SVG ----
			orderedSeries := []testutils.SeriesData{
				*allSeries["Theoretical"],
				*allSeries["Grafite"],
				*allSeries["SNARF"],
				*allSeries["SuRF"],
				*allSeries["SuRFHash(8)"],
				*allSeries["SuRFReal(8)"],
				*allSeries["Adaptive (t=0)"],
				*allSeries["SODA"],
				*allSeries["Hybrid"],
				*allSeries["CDF-ARE"],
				*allSeries["BloomARE"],
			}

			svgPath := fmt.Sprintf("../bench_results/plots/%s/L%d.svg", cfg.distName, rangeLen)
			err := testutils.GenerateTradeoffSVG(
				fmt.Sprintf("FPR vs BPK — %s (60-bit keys, n=%d, L=%d)", cfg.distName, len(cfg.keys), rangeLen),
				"Bits per Key (BPK)",
				"False Positive Rate (FPR)",
				orderedSeries,
				svgPath,
			)
			if err != nil {
				t.Errorf("SVG generation failed: %v", err)
			} else {
				fmt.Printf("\nSVG written to %s\n", svgPath)
			}
		})
	}
}

// --- Distribution-specific tests ---

func TestTradeoff_Clustered(t *testing.T) {
	const (
		n          = 1 << 16
		queryCount = 1 << 18
		nClusters  = 5
		unifFrac   = 0.15
	)

	rng := rand.New(rand.NewSource(99))
	rawKeys, clusters := testutils.GenerateClusterDistribution(n, nClusters, unifFrac, rng)
	keys := mask60Keys(rawKeys)

	runTradeoffBench(t, benchConfig{
		distName: "clustered",
		keys:     keys,
		queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
			qrng := rand.New(rand.NewSource(seed))
			return mask60Queries(testutils.GenerateClusterQueries(queryCount, clusters, unifFrac, rangeLen, qrng))
		},
	})
}

func TestTradeoff_Uniform(t *testing.T) {
	const (
		n          = 1 << 16
		queryCount = 1 << 18
	)

	rng := rand.New(rand.NewSource(42))
	keys := generateUniformKeys(n, rng)

	runTradeoffBench(t, benchConfig{
		distName: "uniform",
		keys:     keys,
		queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
			qrng := rand.New(rand.NewSource(seed))
			return generateUniformQueries(queryCount, rangeLen, qrng)
		},
	})
}

func TestTradeoff_Spread(t *testing.T) {
	const (
		n          = 1 << 16
		queryCount = 1 << 18
	)

	keys := generateSpreadKeys(n)

	runTradeoffBench(t, benchConfig{
		distName: "spread",
		keys:     keys,
		queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
			qrng := rand.New(rand.NewSource(seed))
			return generateUniformQueries(queryCount, rangeLen, qrng)
		},
	})
}

func TestTradeoff_Zipfian(t *testing.T) {
	const (
		n          = 1 << 16
		queryCount = 1 << 18
		nPrefixes  = 100
	)

	rng := rand.New(rand.NewSource(77))
	keys, prefixes := generateZipfianKeys(n, nPrefixes, rng)

	runTradeoffBench(t, benchConfig{
		distName: "zipfian",
		keys:     keys,
		queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
			qrng := rand.New(rand.NewSource(seed))
			return generateZipfianQueries(queryCount, prefixes, rangeLen, qrng)
		},
	})
}

func TestTradeoff_Temporal(t *testing.T) {
	const (
		n          = 1 << 16
		queryCount = 1 << 18
	)

	rng := rand.New(rand.NewSource(55))
	keys := generateTemporalKeys(n, rng)

	runTradeoffBench(t, benchConfig{
		distName: "temporal",
		keys:     keys,
		queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
			qrng := rand.New(rand.NewSource(seed))
			return generateTemporalQueries(queryCount, keys, rangeLen, qrng)
		},
	})
}

// --- Key and query generators ---

func generateUniformKeys(n int, rng *rand.Rand) []uint64 {
	seen := make(map[uint64]bool, n)
	keys := make([]uint64, 0, n)
	for len(keys) < n {
		k := rng.Uint64() & mask60
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys
}

func generateUniformQueries(count int, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	queries := make([][2]uint64, count)
	for i := range queries {
		a := rng.Uint64() & mask60
		queries[i] = [2]uint64{a, a + rangeLen - 1}
	}
	return queries
}

func generateSpreadKeys(n int) []uint64 {
	step := (uint64(1) << 60) / uint64(n)
	keys := make([]uint64, n)
	for i := 0; i < n; i++ {
		keys[i] = uint64(i) * step
	}
	return keys
}

func generateZipfianKeys(n, nPrefixes int, rng *rand.Rand) ([]uint64, []uint64) {
	prefixes := make([]uint64, nPrefixes)
	for i := range prefixes {
		prefixes[i] = rng.Uint64() & ((1 << 40) - 1)
	}
	sort.Slice(prefixes, func(i, j int) bool { return prefixes[i] < prefixes[j] })

	nTop := nPrefixes / 10
	nHot := n * 80 / 100

	seen := make(map[uint64]bool, n)
	keys := make([]uint64, 0, n)
	for len(keys) < nHot {
		pref := prefixes[rng.Intn(nTop)]
		k := (pref << 20) | (rng.Uint64() & ((1 << 20) - 1))
		k &= mask60
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	for len(keys) < n {
		pref := prefixes[nTop+rng.Intn(nPrefixes-nTop)]
		k := (pref << 20) | (rng.Uint64() & ((1 << 20) - 1))
		k &= mask60
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys, prefixes
}

func generateZipfianQueries(count int, prefixes []uint64, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	nTop := len(prefixes) / 10
	queries := make([][2]uint64, count)
	nHotQ := count * 80 / 100
	for i := 0; i < nHotQ; i++ {
		pref := prefixes[rng.Intn(nTop)]
		a := (pref << 20) | (rng.Uint64() & ((1 << 20) - 1))
		a &= mask60
		queries[i] = [2]uint64{a, a + rangeLen - 1}
	}
	for i := nHotQ; i < count; i++ {
		a := rng.Uint64() & mask60
		queries[i] = [2]uint64{a, a + rangeLen - 1}
	}
	return queries
}

func generateTemporalKeys(n int, rng *rand.Rand) []uint64 {
	base := uint64(1) << 50
	step := uint64(1000)
	jitter := float64(step) / 4.0

	raw := make([]uint64, 0, n*2)
	pos := base
	for len(raw) < n*3/2 {
		offset := int64(rng.NormFloat64() * jitter)
		k := uint64(int64(pos) + offset)
		k &= mask60
		raw = append(raw, k)
		pos += step
	}

	// TTL gap: remove early 30% of keys, keep ~10% survivors from that region
	gapEnd := len(raw) * 30 / 100
	survivors := make([]uint64, 0, n)
	survivors = append(survivors, raw[gapEnd:]...)
	for i := 0; i < gapEnd; i++ {
		if rng.Float64() < 0.10 {
			survivors = append(survivors, raw[i])
		}
	}

	seen := make(map[uint64]bool, len(survivors))
	keys := make([]uint64, 0, n)
	for _, k := range survivors {
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	if len(keys) > n {
		keys = keys[:n]
	}
	return keys
}

func generateTemporalQueries(count int, keys []uint64, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	queries := make([][2]uint64, count)
	minK, maxK := keys[0], keys[len(keys)-1]
	spread := maxK - minK
	for i := range queries {
		var a uint64
		if rng.Float64() < 0.5 {
			recentBase := maxK - spread*30/100
			a = recentBase + uint64(rng.Int63n(int64(spread*30/100)))
		} else {
			a = minK + uint64(rng.Int63n(int64(spread)))
		}
		a &= mask60
		queries[i] = [2]uint64{a, a + rangeLen - 1}
	}
	return queries
}

// --- Sanity tests ---

func TestSanity_Grafite(t *testing.T) {
	keys := []uint64{0, 1_000_000_000, 2_000_000_000}
	f := grafite.New(keys, 6.0)
	if f.SizeInBits() == 0 {
		t.Error("expected SizeInBits > 0")
	}
	if f.IsEmpty(0, 1) {
		t.Error("false negative: IsEmpty(0,1) must be false — key 0 is in range")
	}
	if f.IsEmpty(999_999_999, 1_000_000_001) {
		t.Error("false negative: key 1e9 is in range")
	}
}

func TestSanity_SuRF(t *testing.T) {
	keys := []uint64{10, 20, 30}
	f := surf.New(keys, surf.SuffixNone, 0, 0)
	if f.SizeInBits() == 0 {
		t.Error("expected SizeInBits > 0")
	}
	if f.IsEmpty(9, 11) {
		t.Error("false negative: key 10 is in range [9,11]")
	}
	if f.IsEmpty(19, 21) {
		t.Error("false negative: key 20 is in range [19,21]")
	}
}

func TestSanity_SNARF(t *testing.T) {
	keys := []uint64{0, 1_000_000_000, 2_000_000_000}
	f := snarf.New(keys, 6.0)
	if f.SizeInBits() == 0 {
		t.Error("expected SizeInBits > 0")
	}
	if f.IsEmpty(0, 1) {
		t.Error("false negative: key 0 is in range [0,1]")
	}
	if f.IsEmpty(999_999_999, 1_000_000_001) {
		t.Error("false negative: key 1e9 is in range")
	}
}

var _ = are.NewApproximateRangeEmptiness
