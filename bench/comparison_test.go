package bench_test

import (
	"Thesis-bench-industry/grafite"
	"Thesis-bench-industry/snarf"
	"Thesis-bench-industry/surf"
	"Thesis/bits"
	"Thesis/emptiness/are_trunc"
	"Thesis/emptiness/are_bloom"
	"Thesis/emptiness/are_hybrid"
	"Thesis/emptiness/are_hybrid_scan"
	"Thesis/emptiness/are_adaptive"
	"Thesis/emptiness/are_pgm"
	"Thesis/emptiness/are_soda_hash"
	"Thesis/testutils"
	"encoding/binary"
	"encoding/json"
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

// saveSyntheticKeys saves keys in SOSD binary format: [uint64 count LE][count × uint64 keys LE].
func saveSyntheticKeys(path string, keys []uint64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := binary.Write(f, binary.LittleEndian, uint64(len(keys))); err != nil {
		return err
	}
	return binary.Write(f, binary.LittleEndian, keys)
}

// loadSyntheticKeys loads keys from SOSD binary format. Returns error if file doesn't exist.
func loadSyntheticKeys(path string) ([]uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var count uint64
	if err := binary.Read(f, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("read count: %w", err)
	}
	keys := make([]uint64, count)
	if err := binary.Read(f, binary.LittleEndian, keys); err != nil {
		return nil, fmt.Errorf("read keys: %w", err)
	}
	return keys, nil
}

// cacheOrGenerate tries to load keys from a cache file. If not found, calls generate(),
// saves to cache, and returns the keys. Cache path: {cacheDir}/{distName}_{n}.bin.
func cacheOrGenerate(cacheDir, distName string, n int, generate func() []uint64) []uint64 {
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		fmt.Printf("[GEN KEYS] %s n=%d (mkdir failed: %v, generating...)\n", distName, n, err)
		return generate()
	}
	path := fmt.Sprintf("%s/%s_%d.bin", cacheDir, distName, n)
	if keys, err := loadSyntheticKeys(path); err == nil {
		fmt.Printf("[CACHED KEYS] %s n=%d (loaded from %s)\n", distName, n, path)
		return keys
	}
	keys := generate()
	if err := saveSyntheticKeys(path, keys); err != nil {
		fmt.Printf("[GEN KEYS] %s n=%d (save failed: %v)\n", distName, n, err)
	} else {
		fmt.Printf("[GEN KEYS] %s n=%d (saved to %s)\n", distName, n, path)
	}
	return keys
}

// benchConfig parameterises a single distribution benchmark.
type benchConfig struct {
	distName   string
	n          int
	keys       []uint64
	queryCount int
	queryFunc  func(rangeLen uint64, seed int64) [][2]uint64
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

func avgFPRParallel(keys []uint64, queryFunc func(uint64, int64) [][2]uint64, rangeLen uint64, seeds []int64, isEmpty func(a, b uint64) bool) float64 {
	results := make([]float64, len(seeds))
	var wg sync.WaitGroup
	for i, seed := range seeds {
		i, seed := i, seed
		wg.Add(1)
		go func() {
			defer wg.Done()
			qs := queryFunc(rangeLen, seed)
			results[i] = testutils.MeasureFPR(keys, qs, isEmpty)
		}()
	}
	wg.Wait()
	sum := 0.0
	for _, v := range results {
		sum += v
	}
	return sum / float64(len(seeds))
}

func avgFPRSeq(keys []uint64, queryFunc func(uint64, int64) [][2]uint64, rangeLen uint64, seeds []int64, isEmpty func(a, b uint64) bool) float64 {
	sum := 0.0
	for _, seed := range seeds {
		qs := queryFunc(rangeLen, seed)
		sum += testutils.MeasureFPR(keys, qs, isEmpty)
	}
	return sum / float64(len(seeds))
}

type seriesPoint struct {
	series string
	point  testutils.Point
	label  string
}

type savedSeries struct {
	Name   string            `json:"name"`
	Points []testutils.Point `json:"points"`
	Params json.RawMessage   `json:"params,omitempty"`
}

// seriesParamsKGrid holds hyperparameters for K-grid filters.
type seriesParamsKGrid struct {
	Type        string   `json:"type"`
	KGrid       []uint32 `json:"kGrid"`
	RangeLen    uint64   `json:"rangeLen"`
	NKeys       int      `json:"nKeys"`
	QuerySeeds  []int64  `json:"querySeeds"`
	QueryCount  int      `json:"queryCount"`
	NRuns       int      `json:"nRuns"`
}

// seriesParamsEpsilon holds hyperparameters for epsilon-loop filters.
type seriesParamsEpsilon struct {
	Type       string    `json:"type"`
	Epsilons   []float64 `json:"epsilons"`
	RangeLen   uint64    `json:"rangeLen"`
	NKeys      int       `json:"nKeys"`
	QuerySeeds []int64   `json:"querySeeds"`
	QueryCount int       `json:"queryCount"`
	NRuns      int       `json:"nRuns"`
}

// seriesParamsBPKSweep holds hyperparameters for BPK-sweep CGo filters.
type seriesParamsBPKSweep struct {
	Type       string    `json:"type"`
	BPKSweep   []float64 `json:"bpkSweep"`
	RangeLen   uint64    `json:"rangeLen"`
	NKeys      int       `json:"nKeys"`
	QuerySeeds []int64   `json:"querySeeds"`
	QueryCount int       `json:"queryCount"`
	NRuns      int       `json:"nRuns"`
}

// seriesParamsTheoretical holds hyperparameters for the Theoretical series.
type seriesParamsTheoretical struct {
	Type     string   `json:"type"`
	KGrid    []uint32 `json:"kGrid"`
	RangeLen uint64   `json:"rangeLen"`
}

func buildParamsKGrid(kGrid []uint32, rangeLen uint64, nKeys, queryCount int, seeds []int64, nRuns int) json.RawMessage {
	p := seriesParamsKGrid{
		Type:       "kgrid",
		KGrid:      kGrid,
		RangeLen:   rangeLen,
		NKeys:      nKeys,
		QuerySeeds: seeds,
		QueryCount: queryCount,
		NRuns:      nRuns,
	}
	b, _ := json.Marshal(p)
	return b
}

func buildParamsEpsilon(epsilons []float64, rangeLen uint64, nKeys, queryCount int, seeds []int64, nRuns int) json.RawMessage {
	p := seriesParamsEpsilon{
		Type:       "epsilon",
		Epsilons:   epsilons,
		RangeLen:   rangeLen,
		NKeys:      nKeys,
		QuerySeeds: seeds,
		QueryCount: queryCount,
		NRuns:      nRuns,
	}
	b, _ := json.Marshal(p)
	return b
}

func buildParamsBPKSweep(bpkSweep []float64, rangeLen uint64, nKeys, queryCount int, seeds []int64, nRuns int) json.RawMessage {
	p := seriesParamsBPKSweep{
		Type:       "bpksweep",
		BPKSweep:   bpkSweep,
		RangeLen:   rangeLen,
		NKeys:      nKeys,
		QuerySeeds: seeds,
		QueryCount: queryCount,
		NRuns:      nRuns,
	}
	b, _ := json.Marshal(p)
	return b
}

func buildParamsTheoretical(kGrid []uint32, rangeLen uint64) json.RawMessage {
	p := seriesParamsTheoretical{
		Type:     "theoretical",
		KGrid:    kGrid,
		RangeLen: rangeLen,
	}
	b, _ := json.Marshal(p)
	return b
}

// loadCachedSeries loads all saved series from the JSON file, keyed by name.
// Returns an empty map (not an error) when the file does not exist.
func loadCachedSeries(path string) map[string]savedSeries {
	result := make(map[string]savedSeries)
	data, err := os.ReadFile(path)
	if err != nil {
		return result
	}
	var saved []savedSeries
	if err := json.Unmarshal(data, &saved); err != nil {
		return result
	}
	for _, s := range saved {
		result[s.Name] = s
	}
	return result
}

// mergePoints merges newPoints into existing points (dedup by X, keep newest value).
// Old points whose X is not in newPoints are preserved unchanged.
func mergePoints(existing, newPoints []testutils.Point) []testutils.Point {
	byX := make(map[float64]float64, len(existing))
	// Start with existing points.
	for _, p := range existing {
		byX[p.X] = p.Y
	}
	// New points overwrite existing ones at the same X.
	for _, p := range newPoints {
		byX[p.X] = p.Y
	}
	merged := make([]testutils.Point, 0, len(byX))
	for x, y := range byX {
		merged = append(merged, testutils.Point{X: x, Y: y})
	}
	sort.Slice(merged, func(i, j int) bool { return merged[i].X < merged[j].X })
	return merged
}

// saveSeriesDataWithCache merges new series data into existing cached data and writes to disk.
// For each series in newSeries: if it has points, merge them with existing cached points.
// Cached series that are not in newSeries are preserved as-is.
// The params field is set to the provided params map (keyed by series name).
func saveSeriesDataWithCache(path string, cached map[string]savedSeries, newSeries map[string]*testutils.SeriesData, newParams map[string]json.RawMessage) error {
	// Build output: start from cached, overlay new data.
	out := make(map[string]savedSeries, len(cached))
	for name, s := range cached {
		out[name] = s
	}
	for name, s := range newSeries {
		existing := out[name]
		merged := mergePoints(existing.Points, s.Points)
		params := existing.Params
		if p, ok := newParams[name]; ok {
			params = p
		}
		out[name] = savedSeries{Name: name, Points: merged, Params: params}
	}

	// Emit as a sorted slice for stable output.
	result := make([]savedSeries, 0, len(out))
	for _, s := range out {
		result = append(result, s)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })

	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func saveSeriesData(path string, series map[string]*testutils.SeriesData) error {
	var out []savedSeries
	for _, s := range series {
		if len(s.Points) > 0 {
			pts := make([]testutils.Point, len(s.Points))
			copy(pts, s.Points)
			sort.Slice(pts, func(i, j int) bool { return pts[i].X < pts[j].X })
			// Deduplicate by X (keep first occurrence)
			deduped := pts[:1]
			for _, p := range pts[1:] {
				if p.X != deduped[len(deduped)-1].X {
					deduped = append(deduped, p)
				}
			}
			out = append(out, savedSeries{Name: s.Name, Points: deduped})
		}
	}
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func loadSeriesData(path string, series map[string]*testutils.SeriesData) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var saved []savedSeries
	if err := json.Unmarshal(data, &saved); err != nil {
		return err
	}
	for _, s := range saved {
		if target, ok := series[s.Name]; ok {
			target.Points = s.Points
		}
	}
	return nil
}

// parseEnvSet parses a comma-separated env var into a set of names.
// Returns nil if the env var is empty.
func parseEnvSet(envVar string) map[string]bool {
	val := os.Getenv(envVar)
	if val == "" {
		return nil
	}
	set := make(map[string]bool)
	for _, name := range strings.Split(val, ",") {
		name = strings.TrimSpace(name)
		if name != "" {
			set[name] = true
		}
	}
	return set
}

// paramsEqual returns true if two json.RawMessage values encode the same JSON
// (byte-for-byte identical after marshalling — only valid since we always
// marshal from the same struct types in the same field order).
func paramsEqual(a, b json.RawMessage) bool {
	if a == nil || b == nil {
		return false
	}
	return string(a) == string(b)
}

// shouldSkipSeries decides whether a series should be skipped (use cache).
// Returns (skip bool, reason string).
func shouldSkipSeries(name string, onlySet, skipSet map[string]bool, cached map[string]savedSeries, currentParams json.RawMessage) (bool, string) {
	// SKIP env var takes priority.
	if skipSet != nil && skipSet[name] {
		return true, "SKIP env var"
	}
	// ONLY env var: skip everything not in the set.
	if onlySet != nil && !onlySet[name] {
		return true, "ONLY env var"
	}
	// Params-based: skip if cached params match current params.
	if cs, ok := cached[name]; ok && len(cs.Points) > 0 {
		if paramsEqual(cs.Params, currentParams) {
			return true, fmt.Sprintf("params match, %d points", len(cs.Points))
		}
	}
	return false, ""
}

func runTradeoffBench(t *testing.T, cfg benchConfig) {
	const nRuns = 3

	rangeLens := []uint64{1, 16, 128, 1024, 4096, 16384, 65536}
	kGrid := []uint32{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 28, 32, 36, 40, 48}
	bpkSweep := []float64{4, 6, 8, 10, 12, 14, 16, 18, 20}
	epsilons := []float64{0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001}

	keysBS := make([]bits.BitString, len(cfg.keys))
	for i, v := range cfg.keys {
		keysBS[i] = testutils.TrieBS(v)
	}

	os.MkdirAll(fmt.Sprintf("../bench_results/plots/N%d/%s", cfg.n, cfg.distName), 0755)

	// Parse ONLY/SKIP env vars once (shared across all range lengths).
	onlySet := parseEnvSet("ONLY")
	skipSet := parseEnvSet("SKIP")

	for _, rangeLen := range rangeLens {
		t.Run(fmt.Sprintf("L=%d", rangeLen), func(t *testing.T) {
			// ---- series map ----
			allSeries := map[string]*testutils.SeriesData{
				"Theoretical":    {Name: "Theoretical", Color: "#ef4444", Dashed: true, Marker: "circle"},
				"Grafite":        {Name: "Grafite", Color: "#1a6b3c", Marker: "diamond"},
				"SNARF":          {Name: "SNARF", Color: "#1a3a6b", Marker: "star"},
				"SuRF":           {Name: "SuRF", Color: "#111111", Marker: "square"},
				"SuRFHash(8)":    {Name: "SuRFHash(8)", Color: "#111111", Marker: "triangle"},
				"SuRFReal(8)":    {Name: "SuRFReal(8)", Color: "#111111", Marker: "diamond"},
				"Truncation":     {Name: "Truncation", Color: "#9b59b6", Marker: "triangle"},
				"Adaptive (t=0)": {Name: "Adaptive (t=0)", Color: "#2a7fff", Marker: "square"},
				"SODA":           {Name: "SODA", Color: "#4dd88a", Marker: "diamond"},
				"Hybrid":         {Name: "Hybrid", Color: "#ff6b6b", Marker: "star"},
				"Scan-ARE":       {Name: "Scan-ARE", Color: "#06b6d4", Marker: "star"},
				"CDF-ARE":        {Name: "CDF-ARE", Color: "#ff922b", Marker: "circle"},
				"BloomARE":       {Name: "BloomARE", Color: "#888888", Dashed: true, Marker: "circle"},
			}

			dataDir := fmt.Sprintf("../bench_results/data/N%d/%s", cfg.n, cfg.distName)
			os.MkdirAll(dataDir, 0755)
			dataPath := fmt.Sprintf("%s/L%d.json", dataDir, rangeLen)
			plotOnly := os.Getenv("PLOT_ONLY") != ""

			if plotOnly {
				if err := loadSeriesData(dataPath, allSeries); err != nil {
					t.Skipf("no saved data for %s/L%d: %v", cfg.distName, rangeLen, err)
					return
				}
				fmt.Printf("\n=== Plot-only mode — %s L=%d (loaded from %s) ===\n", cfg.distName, rangeLen, dataPath)
			} else {
				seeds := []int64{12345, 54321, 99999}

				// Load existing cache for per-series skip logic.
				cached := loadCachedSeries(dataPath)

				// Pre-compute current params for each series group.
				paramsKGrid := buildParamsKGrid(kGrid, rangeLen, len(cfg.keys), cfg.queryCount, seeds, nRuns)
				paramsEpsilon := buildParamsEpsilon(epsilons, rangeLen, len(cfg.keys), cfg.queryCount, seeds, nRuns)
				paramsBPKSweep := buildParamsBPKSweep(bpkSweep, rangeLen, len(cfg.keys), cfg.queryCount, seeds, nRuns)
				paramsTheoretical := buildParamsTheoretical(kGrid, rangeLen)

				// Determine per-series params mapping (used for saving).
				seriesParams := map[string]json.RawMessage{
					"Theoretical":    paramsTheoretical,
					"Truncation":     paramsKGrid,
					"Adaptive (t=0)": paramsKGrid,
					"SODA":           paramsKGrid,
					"Hybrid":         paramsKGrid,
					"Scan-ARE":       paramsKGrid,
					"CDF-ARE":        paramsEpsilon,
					"BloomARE":       paramsEpsilon,
					"Grafite":        paramsBPKSweep,
					"SNARF":          paramsBPKSweep,
					"SuRF":           paramsBPKSweep,
					"SuRFHash(8)":    paramsBPKSweep,
					"SuRFReal(8)":    paramsBPKSweep,
				}

				// newSeriesParams tracks which params to record for rebuilt series.
				newParams := make(map[string]json.RawMessage)

				// Restore cached points for all series upfront (will be overwritten if rebuilt).
				for name, cs := range cached {
					if sd, ok := allSeries[name]; ok {
						sd.Points = cs.Points
					}
				}

				fmt.Printf("\n=== Industry Comparison — %s (60-bit keys, %d keys, L=%d, %d runs) ===\n", cfg.distName, len(cfg.keys), rangeLen, nRuns)
				fmt.Printf("%-16s | %8s | %14s\n", "Series", "BPK", "FPR(avg)")
				fmt.Println(strings.Repeat("-", 45))

				// Helper: decide skip and log.
				type skipDecision struct {
					skip   bool
					reason string
				}
				decideSkip := func(name string, params json.RawMessage) skipDecision {
					skip, reason := shouldSkipSeries(name, onlySet, skipSet, cached, params)
					if skip {
						fmt.Printf("[CACHED] %-16s (%s)\n", name, reason)
					} else {
						fmt.Printf("[BUILD]  %-16s (params changed)\n", name)
						newParams[name] = seriesParams[name]
					}
					return skipDecision{skip, reason}
				}

				// ---- Theoretical (derived from K-grid) ----
				if d := decideSkip("Theoretical", paramsTheoretical); !d.skip {
					allSeries["Theoretical"].Points = nil
					for _, K := range kGrid {
						thEps := float64(rangeLen) / math.Exp2(float64(K))
						if thEps > 0 && thEps <= 1 {
							allSeries["Theoretical"].Points = append(allSeries["Theoretical"].Points,
								testutils.Point{X: float64(K), Y: thEps})
						}
					}
				}

				// ---- Build & measure ARE filters in parallel (pure Go, thread-safe) ----
				type fprTask struct {
					series  string
					label   string
					bpk     float64
					isEmpty func(a, b uint64) bool
				}
				var goTasks []fprTask

				// Determine which K-grid series to rebuild (logs [CACHED]/[BUILD] once per series).
				kgridSeriesNames := []string{"Truncation", "Adaptive (t=0)", "SODA", "Hybrid", "Scan-ARE"}
				rebuildKGridSeries := make(map[string]bool)
				for _, name := range kgridSeriesNames {
					if d := decideSkip(name, paramsKGrid); !d.skip {
						rebuildKGridSeries[name] = true
					}
				}
				rebuildKGrid := len(rebuildKGridSeries) > 0

				if rebuildKGrid {
					// Clear points for series that need rebuilding.
					for name := range rebuildKGridSeries {
						allSeries[name].Points = nil
					}

					for _, K := range kGrid {
						K := K
						if rebuildKGridSeries["Truncation"] {
							if f, err := are_trunc.NewApproximateRangeEmptinessFromK(keysBS, K); err == nil {
								bpk := float64(f.SizeInBits()) / float64(len(cfg.keys))
								goTasks = append(goTasks, fprTask{"Truncation", fmt.Sprintf("Truncation(K=%d)", K), bpk,
									func(a, b uint64) bool { return f.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) }})
							}
						}
						if rebuildKGridSeries["Adaptive (t=0)"] {
							if f, err := are_adaptive.NewAdaptiveAREFromK(keysBS, rangeLen, K, 0); err == nil {
								bpk := float64(f.SizeInBits()) / float64(len(cfg.keys))
								goTasks = append(goTasks, fprTask{"Adaptive (t=0)", fmt.Sprintf("Adaptive(K=%d)", K), bpk,
									func(a, b uint64) bool { return f.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) }})
							}
						}
						if rebuildKGridSeries["SODA"] {
							if f, err := are_soda_hash.NewApproximateRangeEmptinessSodaFromK(cfg.keys, rangeLen, K); err == nil {
								bpk := float64(f.SizeInBits()) / float64(len(cfg.keys))
								goTasks = append(goTasks, fprTask{"SODA", fmt.Sprintf("SODA(K=%d)", K), bpk,
									func(a, b uint64) bool { return f.IsEmpty(a, b) }})
							}
						}
						if rebuildKGridSeries["Hybrid"] {
							if f, err := are_hybrid.NewHybridAREFromK(keysBS, rangeLen, K); err == nil {
								bpk := float64(f.SizeInBits()) / float64(len(cfg.keys))
								goTasks = append(goTasks, fprTask{"Hybrid", fmt.Sprintf("Hybrid(K=%d)", K), bpk,
									func(a, b uint64) bool { return f.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) }})
							}
						}
						if rebuildKGridSeries["Scan-ARE"] {
							if f, err := are_hybrid_scan.NewHybridScanAREFromK(keysBS, rangeLen, K); err == nil {
								bpk := float64(f.SizeInBits()) / float64(len(cfg.keys))
								goTasks = append(goTasks, fprTask{"Scan-ARE", fmt.Sprintf("Scan-ARE(K=%d)", K), bpk,
									func(a, b uint64) bool { return f.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) }})
							}
						}
					}

					goResults := make([]seriesPoint, len(goTasks))
					var wg sync.WaitGroup
					for i, task := range goTasks {
						i, task := i, task
						wg.Add(1)
						go func() {
							defer wg.Done()
							fpr := avgFPRParallel(cfg.keys, cfg.queryFunc, rangeLen, seeds, task.isEmpty)
							goResults[i] = seriesPoint{task.series, testutils.Point{X: task.bpk, Y: fpr}, task.label}
						}()
					}
					wg.Wait()

					for _, sp := range goResults {
						allSeries[sp.series].Points = append(allSeries[sp.series].Points, sp.point)
						fmt.Printf("%-16s | %8.2f | %14.6f\n", sp.label, sp.point.X, sp.point.Y)
					}
				}

				// ---- Epsilon-loop filters (CDF-ARE, BloomARE) ----
				rebuildEpsilonSeries := make(map[string]bool)
				for _, name := range []string{"CDF-ARE", "BloomARE"} {
					if d := decideSkip(name, paramsEpsilon); !d.skip {
						rebuildEpsilonSeries[name] = true
					}
				}

				if len(rebuildEpsilonSeries) > 0 {
					for name := range rebuildEpsilonSeries {
						allSeries[name].Points = nil
					}

					var epsilonTasks []fprTask
					for _, eps := range epsilons {
						if rebuildEpsilonSeries["CDF-ARE"] {
							if f, err := are_pgm.NewPGMApproximateRangeEmptiness(cfg.keys, rangeLen, eps, 64); err == nil {
								bpk := float64(f.TotalSizeInBits()) / float64(len(cfg.keys))
								epsilonTasks = append(epsilonTasks, fprTask{"CDF-ARE", "CDF-ARE", bpk,
									func(a, b uint64) bool { return f.IsEmpty(a, b) }})
							}
						}
						if rebuildEpsilonSeries["BloomARE"] {
							if f, err := are_bloom.NewBloomARE(cfg.keys, rangeLen, eps); err == nil {
								bpk := float64(f.SizeInBits()) / float64(len(cfg.keys))
								epsilonTasks = append(epsilonTasks, fprTask{"BloomARE", "BloomARE", bpk,
									func(a, b uint64) bool { return f.IsEmpty(a, b) }})
							}
						}
					}

					epsilonResults := make([]seriesPoint, len(epsilonTasks))
					var wg sync.WaitGroup
					for i, task := range epsilonTasks {
						i, task := i, task
						wg.Add(1)
						go func() {
							defer wg.Done()
							fpr := avgFPRParallel(cfg.keys, cfg.queryFunc, rangeLen, seeds, task.isEmpty)
							epsilonResults[i] = seriesPoint{task.series, testutils.Point{X: task.bpk, Y: fpr}, task.label}
						}()
					}
					wg.Wait()

					for _, sp := range epsilonResults {
						allSeries[sp.series].Points = append(allSeries[sp.series].Points, sp.point)
						fmt.Printf("%-16s | %8.2f | %14.6f\n", sp.label, sp.point.X, sp.point.Y)
					}
				}

				// ---- CGo filters: build & measure sequentially (not thread-safe) ----
				// Determine which CGo series to rebuild.
				cgoSeries := []string{"Grafite", "SNARF", "SuRF", "SuRFHash(8)", "SuRFReal(8)"}
				rebuildCGoSeries := make(map[string]bool)
				for _, name := range cgoSeries {
					if d := decideSkip(name, paramsBPKSweep); !d.skip {
						rebuildCGoSeries[name] = true
					}
				}

				if len(rebuildCGoSeries) > 0 {
					for name := range rebuildCGoSeries {
						allSeries[name].Points = nil
					}

					for _, bpk := range bpkSweep {
						if rebuildCGoSeries["Grafite"] {
							if f := tryGrafite(cfg.keys, bpk); f != nil {
								actualBPK := float64(f.SizeInBits()) / float64(len(cfg.keys))
								fpr := avgFPRSeq(cfg.keys, cfg.queryFunc, rangeLen, seeds, func(a, b uint64) bool { return f.IsEmpty(a, b) })
								allSeries["Grafite"].Points = append(allSeries["Grafite"].Points,
									testutils.Point{X: actualBPK, Y: fpr})
								fmt.Printf("%-16s | %8.2f | %14.6f\n", fmt.Sprintf("Grafite(bpk=%.0f)", bpk), actualBPK, fpr)
							}
						}

						if rebuildCGoSeries["SNARF"] {
							f := snarf.New(cfg.keys, bpk)
							actualBPK := float64(f.SizeInBits()) / float64(len(cfg.keys))
							fpr := avgFPRSeq(cfg.keys, cfg.queryFunc, rangeLen, seeds, func(a, b uint64) bool { return f.IsEmpty(a, b) })
							allSeries["SNARF"].Points = append(allSeries["SNARF"].Points,
								testutils.Point{X: actualBPK, Y: fpr})
							fmt.Printf("%-16s | %8.2f | %14.6f\n", fmt.Sprintf("SNARF(bpk=%.0f)", bpk), actualBPK, fpr)
						}
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
						if rebuildCGoSeries[sv.name] {
							f := surf.New(cfg.keys, sv.st, sv.hashBits, sv.realBits)
							actualBPK := float64(f.SizeInBits()) / float64(len(cfg.keys))
							fpr := avgFPRSeq(cfg.keys, cfg.queryFunc, rangeLen, seeds, func(a, b uint64) bool { return f.IsEmpty(a, b) })
							allSeries[sv.name].Points = append(allSeries[sv.name].Points,
								testutils.Point{X: actualBPK, Y: fpr})
							fmt.Printf("%-16s | %8.2f | %14.6f\n", sv.name, actualBPK, fpr)
						}
					}
				}

				// Save: merge new data into cache, keep all old points.
				// Build a map of only the series that were rebuilt (new points).
				rebuiltSeries := make(map[string]*testutils.SeriesData)
				for name := range newParams {
					rebuiltSeries[name] = allSeries[name]
				}
				if err := saveSeriesDataWithCache(dataPath, cached, rebuiltSeries, newParams); err != nil {
					t.Logf("warning: failed to save data: %v", err)
				} else {
					// Reload cache so allSeries reflects the merged state for plotting.
					merged := loadCachedSeries(dataPath)
					for name, cs := range merged {
						if sd, ok := allSeries[name]; ok {
							sd.Points = cs.Points
						}
					}
				}

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
				*allSeries["Truncation"],
				*allSeries["SODA"],
				*allSeries["Hybrid"],
				*allSeries["Scan-ARE"],
				*allSeries["CDF-ARE"],
				*allSeries["BloomARE"],
			}

			svgPath := fmt.Sprintf("../bench_results/plots/N%d/%s/L%d.svg", cfg.n, cfg.distName, rangeLen)
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

// clusterMeta is the JSON-serialisable form of []testutils.ClusterInfo.
type clusterMeta struct {
	Center uint64  `json:"center"`
	Stddev float64 `json:"stddev"`
}

func TestTradeoff_Clustered(t *testing.T) {
	const (
		queryCount = 1 << 18
		nClusters  = 5
		unifFrac   = 0.15
		cacheDir   = "../bench/synthetic_data"
	)
	for _, n := range []int{1 << 16, 1 << 18, 1 << 20} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			keysPath := fmt.Sprintf("%s/clustered_%d.bin", cacheDir, n)
			metaPath := fmt.Sprintf("%s/clustered_%d_meta.json", cacheDir, n)

			os.MkdirAll(cacheDir, 0755)

			var keys []uint64
			var clusters []testutils.ClusterInfo

			cachedKeys, keyErr := loadSyntheticKeys(keysPath)
			metaBytes, metaErr := os.ReadFile(metaPath)

			if keyErr == nil && metaErr == nil {
				var meta []clusterMeta
				if json.Unmarshal(metaBytes, &meta) == nil {
					clusters = make([]testutils.ClusterInfo, len(meta))
					for i, m := range meta {
						clusters[i] = testutils.ClusterInfo{Center: m.Center, Stddev: m.Stddev}
					}
					keys = cachedKeys
					fmt.Printf("[CACHED KEYS] clustered n=%d (loaded from %s)\n", n, keysPath)
				}
			}

			if keys == nil {
				rng := rand.New(rand.NewSource(99))
				rawKeys, cls := testutils.GenerateClusterDistribution(n, nClusters, unifFrac, rng)
				keys = mask60Keys(rawKeys)
				clusters = cls

				if err := saveSyntheticKeys(keysPath, keys); err != nil {
					fmt.Printf("[GEN KEYS] clustered n=%d (key save failed: %v)\n", n, err)
				} else {
					meta := make([]clusterMeta, len(clusters))
					for i, c := range clusters {
						meta[i] = clusterMeta{Center: c.Center, Stddev: c.Stddev}
					}
					if b, err := json.MarshalIndent(meta, "", "  "); err == nil {
						if err := os.WriteFile(metaPath, b, 0644); err != nil {
							fmt.Printf("[GEN KEYS] clustered n=%d (meta save failed: %v)\n", n, err)
						} else {
							fmt.Printf("[GEN KEYS] clustered n=%d (saved to %s)\n", n, keysPath)
						}
					}
				}
			}

			runTradeoffBench(t, benchConfig{
				distName:   "clustered",
				n:          n,
				keys:       keys,
				queryCount: queryCount,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					qrng := rand.New(rand.NewSource(seed))
					return mask60Queries(testutils.GenerateClusterQueries(queryCount, clusters, unifFrac, rangeLen, qrng))
				},
			})
		})
	}
}

func TestTradeoff_Uniform(t *testing.T) {
	const queryCount = 1 << 18
	for _, n := range []int{1 << 16, 1 << 18, 1 << 20} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			keys := cacheOrGenerate("../bench/synthetic_data", "uniform", n, func() []uint64 {
				rng := rand.New(rand.NewSource(42))
				return generateUniformKeys(n, rng)
			})
			runTradeoffBench(t, benchConfig{
				distName:   "uniform",
				n:          n,
				keys:       keys,
				queryCount: queryCount,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					qrng := rand.New(rand.NewSource(seed))
					return generateUniformQueries(queryCount, rangeLen, qrng)
				},
			})
		})
	}
}

func TestTradeoff_Spread(t *testing.T) {
	const queryCount = 1 << 18
	for _, n := range []int{1 << 16, 1 << 18, 1 << 20} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			keys := cacheOrGenerate("../bench/synthetic_data", "spread", n, func() []uint64 {
				return generateSpreadKeys(n)
			})
			runTradeoffBench(t, benchConfig{
				distName:   "spread",
				n:          n,
				keys:       keys,
				queryCount: queryCount,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					qrng := rand.New(rand.NewSource(seed))
					return generateUniformQueries(queryCount, rangeLen, qrng)
				},
			})
		})
	}
}

func TestTradeoff_Zipfian(t *testing.T) {
	const (
		queryCount = 1 << 18
		nPrefixes  = 100
		cacheDir   = "../bench/synthetic_data"
	)
	for _, n := range []int{1 << 16, 1 << 18, 1 << 20} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			keysPath := fmt.Sprintf("%s/zipfian_%d.bin", cacheDir, n)
			prefixesPath := fmt.Sprintf("%s/zipfian_%d_prefixes.bin", cacheDir, n)

			os.MkdirAll(cacheDir, 0755)

			var keys, prefixes []uint64

			cachedKeys, keyErr := loadSyntheticKeys(keysPath)
			cachedPrefixes, prefixErr := loadSyntheticKeys(prefixesPath)

			if keyErr == nil && prefixErr == nil {
				keys = cachedKeys
				prefixes = cachedPrefixes
				fmt.Printf("[CACHED KEYS] zipfian n=%d (loaded from %s)\n", n, keysPath)
			} else {
				rng := rand.New(rand.NewSource(77))
				keys, prefixes = generateZipfianKeys(n, nPrefixes, rng)

				saveErr := saveSyntheticKeys(keysPath, keys)
				if saveErr != nil {
					fmt.Printf("[GEN KEYS] zipfian n=%d (key save failed: %v)\n", n, saveErr)
				} else if err := saveSyntheticKeys(prefixesPath, prefixes); err != nil {
					fmt.Printf("[GEN KEYS] zipfian n=%d (prefix save failed: %v)\n", n, err)
				} else {
					fmt.Printf("[GEN KEYS] zipfian n=%d (saved to %s)\n", n, keysPath)
				}
			}

			runTradeoffBench(t, benchConfig{
				distName:   "zipfian",
				n:          n,
				keys:       keys,
				queryCount: queryCount,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					qrng := rand.New(rand.NewSource(seed))
					return generateZipfianQueries(queryCount, prefixes, rangeLen, qrng)
				},
			})
		})
	}
}

func TestTradeoff_Temporal(t *testing.T) {
	const queryCount = 1 << 18
	for _, n := range []int{1 << 16, 1 << 18, 1 << 20} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			keys := cacheOrGenerate("../bench/synthetic_data", "temporal", n, func() []uint64 {
				rng := rand.New(rand.NewSource(55))
				return generateTemporalKeys(n, rng)
			})
			runTradeoffBench(t, benchConfig{
				distName:   "temporal",
				n:          n,
				keys:       keys,
				queryCount: queryCount,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					qrng := rand.New(rand.NewSource(seed))
					return generateTemporalQueries(queryCount, keys, rangeLen, qrng)
				},
			})
		})
	}
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

// --- Distribution visualization ---

func normalizedCDF(keys []uint64, sampleEvery int) []testutils.Point {
	n := len(keys)
	minK, maxK := float64(keys[0]), float64(keys[n-1])
	span := maxK - minK
	if span == 0 {
		span = 1
	}
	pts := make([]testutils.Point, 0, n/sampleEvery+2)
	pts = append(pts, testutils.Point{X: 0, Y: 0})
	for i := 0; i < n; i += sampleEvery {
		x := (float64(keys[i]) - minK) / span
		y := float64(i+1) / float64(n)
		pts = append(pts, testutils.Point{X: x, Y: y})
	}
	pts = append(pts, testutils.Point{X: 1, Y: 1})
	return pts
}

func histogram(keys []uint64, nBins int) []testutils.Point {
	n := len(keys)
	minK, maxK := float64(keys[0]), float64(keys[n-1])
	span := maxK - minK
	if span == 0 {
		span = 1
	}
	counts := make([]int, nBins)
	for _, k := range keys {
		bin := int((float64(k) - minK) / span * float64(nBins))
		if bin >= nBins {
			bin = nBins - 1
		}
		counts[bin]++
	}
	maxCount := 0
	for _, c := range counts {
		if c > maxCount {
			maxCount = c
		}
	}
	pts := make([]testutils.Point, nBins)
	for i, c := range counts {
		pts[i] = testutils.Point{
			X: (float64(i) + 0.5) / float64(nBins),
			Y: float64(c) / float64(maxCount),
		}
	}
	return pts
}

func TestDistributionVisualization(t *testing.T) {
	const n = 1 << 16

	type distInfo struct {
		name  string
		keys  []uint64
		color string
	}

	dists := []distInfo{
		{"clustered", func() []uint64 {
			rng := rand.New(rand.NewSource(99))
			raw, _ := testutils.GenerateClusterDistribution(n, 5, 0.15, rng)
			return mask60Keys(raw)
		}(), "#2a7fff"},
		{"uniform", generateUniformKeys(n, rand.New(rand.NewSource(42))), "#22a06b"},
		{"spread", generateSpreadKeys(n), "#e05d10"},
		{"zipfian", func() []uint64 {
			rng := rand.New(rand.NewSource(77))
			keys, _ := generateZipfianKeys(n, 100, rng)
			return keys
		}(), "#9b59b6"},
		{"temporal", generateTemporalKeys(n, rand.New(rand.NewSource(55))), "#c0392b"},
	}

	os.MkdirAll("../bench_results/plots/distributions", 0755)

	// Combined CDF plot
	var cdfSeries []testutils.SeriesData
	for _, d := range dists {
		cdfSeries = append(cdfSeries, testutils.SeriesData{
			Name:   d.name,
			Color:  d.color,
			Marker: "none",
			Points: normalizedCDF(d.keys, 256),
		})
	}
	err := testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("CDF of Key Distributions (n=%d, normalized)", n),
		XLabel: "Normalized Key Position",
		YLabel: "Cumulative Fraction",
	}, cdfSeries, "../bench_results/plots/distributions/cdf_all.svg")
	if err != nil {
		t.Errorf("CDF SVG failed: %v", err)
	} else {
		fmt.Println("CDF written to ../bench_results/plots/distributions/cdf_all.svg")
	}

	// Combined histogram plot
	var histAllSeries []testutils.SeriesData
	for _, d := range dists {
		histAllSeries = append(histAllSeries, testutils.SeriesData{
			Name:   d.name,
			Color:  d.color,
			Marker: "none",
			Points: histogram(d.keys, 200),
		})
	}
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("Key Density — All Distributions (n=%d, 200 bins)", n),
		XLabel: "Normalized Key Position",
		YLabel: "Relative Density",
	}, histAllSeries, "../bench_results/plots/distributions/hist_all.svg")
	if err != nil {
		t.Errorf("combined histogram SVG failed: %v", err)
	} else {
		fmt.Println("Combined histogram written to ../bench_results/plots/distributions/hist_all.svg")
	}

	// Individual histogram per distribution
	for _, d := range dists {
		histSeries := []testutils.SeriesData{{
			Name:   d.name,
			Color:  d.color,
			Marker: "none",
			Points: histogram(d.keys, 200),
		}}
		path := fmt.Sprintf("../bench_results/plots/distributions/hist_%s.svg", d.name)
		err := testutils.GeneratePerformanceSVG(testutils.PlotConfig{
			Title:  fmt.Sprintf("Key Density — %s (n=%d, 200 bins)", d.name, n),
			XLabel: "Normalized Key Position",
			YLabel: "Relative Density",
		}, histSeries, path)
		if err != nil {
			t.Errorf("histogram SVG failed for %s: %v", d.name, err)
		} else {
			fmt.Printf("Histogram written to %s\n", path)
		}
	}
}

