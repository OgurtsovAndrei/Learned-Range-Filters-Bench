package bench_test

import (
	"Thesis-bench-industry/grafite"
	"Thesis/testutils"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"sync"
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

func avgFPRBatch(keys []uint64, queryFunc func(uint64, int64) [][2]uint64, rangeLen uint64, seeds []int64, queryBatch func([][2]uint64) []bool) float64 {
	sum := 0.0
	for _, seed := range seeds {
		qs := queryFunc(rangeLen, seed)
		sum += testutils.MeasureFPRBatch(keys, qs, queryBatch)
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
