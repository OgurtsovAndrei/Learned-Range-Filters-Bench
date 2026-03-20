package bench_test

import (
	"Thesis-bench-industry/grafite"
	"Thesis/testutils"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
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

	// v2 metadata
	keySource           string                 // "synthetic" | "sosd"
	keyFile             string                 // relative path to key file (optional)
	keySeed             *int64                 // RNG seed used for key generation (nil for SOSD)
	keyGenParams        map[string]interface{} // distribution-specific generation params
	queryStrategy       string                 // "cluster" | "uniform" | "zipfian" | "temporal" | "smart_mix"
	queryStrategyParams map[string]interface{} // e.g. smart_mix weights
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
// Supports both v1 (top-level array) and v2 (object with "version") formats.
// Returns an empty map (not an error) when the file does not exist.
func loadCachedSeries(path string) map[string]savedSeries {
	result := make(map[string]savedSeries)
	data, err := os.ReadFile(path)
	if err != nil {
		return result
	}
	data = bytes.TrimSpace(data)
	if len(data) == 0 {
		return result
	}

	// v2: object starting with '{'
	if data[0] == '{' {
		var br benchResult
		if json.Unmarshal(data, &br) != nil {
			return result
		}
		for _, rs := range br.Series {
			ss := savedSeries{Name: rs.Name}
			for _, p := range rs.Points {
				ss.Points = append(ss.Points, testutils.Point{X: p.BPK, Y: p.FPR})
			}
			// Use stored params directly (v1-compatible JSON for shouldSkipSeries).
			ss.Params = rs.Params
			result[rs.Name] = ss
		}
		return result
	}

	// v1: array starting with '['
	var saved []savedSeries
	if json.Unmarshal(data, &saved) != nil {
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
	data = bytes.TrimSpace(data)
	if len(data) == 0 {
		return fmt.Errorf("empty file")
	}

	// v2: object
	if data[0] == '{' {
		var br benchResult
		if err := json.Unmarshal(data, &br); err != nil {
			return err
		}
		for _, rs := range br.Series {
			if target, ok := series[rs.Name]; ok {
				target.Points = nil
				for _, p := range rs.Points {
					target.Points = append(target.Points, testutils.Point{X: p.BPK, Y: p.FPR})
				}
			}
		}
		return nil
	}

	// v1: array
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
	var va, vb interface{}
	if json.Unmarshal(a, &va) != nil || json.Unmarshal(b, &vb) != nil {
		return false
	}
	ca, _ := json.Marshal(va)
	cb, _ := json.Marshal(vb)
	return string(ca) == string(cb)
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

// ---- v2 JSON format ----

type benchResult struct {
	Version   int          `json:"version"`
	Benchmark benchMeta    `json:"benchmark"`
	Keys      keysMeta     `json:"keys"`
	Queries   queriesMeta  `json:"queries"`
	Series    []richSeries `json:"series"`
}

type benchMeta struct {
	Type         string `json:"type"`
	Distribution string `json:"distribution"`
	NKeys        int    `json:"nKeys"`
	RangeLen     uint64 `json:"rangeLen,omitempty"`
	Timestamp    string `json:"timestamp"`
	GitCommit    string `json:"gitCommit"`
}

type keysMeta struct {
	Source           string                 `json:"source"`
	File             string                 `json:"file,omitempty"`
	Seed             *int64                 `json:"seed,omitempty"`
	GenerationParams map[string]interface{} `json:"generationParams,omitempty"`
	Count            int                    `json:"count"`
	SHA256           string                 `json:"sha256"`
}

type queriesMeta struct {
	Strategy       string                 `json:"strategy"`
	StrategyParams map[string]interface{} `json:"strategyParams,omitempty"`
	Count          int                    `json:"count"`
	Seeds          []int64                `json:"seeds"`
	NRuns          int                    `json:"nRuns"`
}

type richSeries struct {
	Name         string          `json:"name"`
	FilterFamily string          `json:"filterFamily"`
	SweepValues  interface{}     `json:"sweepValues"`
	Params       json.RawMessage `json:"params,omitempty"`
	Points       []richPoint     `json:"points"`
}

type richPoint struct {
	SweepParam       float64                `json:"sweepParam"`
	BPK              float64                `json:"bpk"`
	FPR              float64                `json:"fpr"`
	FilterSizeBits   uint64                 `json:"filterSizeBits"`
	BuildTimeNs      *int64                 `json:"buildTimeNs,omitempty"`
	QueryTimeNsPerOp *float64               `json:"queryTimeNsPerOp,omitempty"`
	FilterStats      map[string]interface{} `json:"filterStats,omitempty"`
}

func sha256Keys(keys []uint64) string {
	h := sha256.New()
	buf := make([]byte, 8)
	for _, k := range keys {
		binary.LittleEndian.PutUint64(buf, k)
		h.Write(buf)
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func gitCommitShort() string {
	out, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

func saveBenchResult(path string, result *benchResult) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// loadBenchResult detects v1 (top-level array) vs v2 (object with "version")
// and loads either format. v1 files are converted to benchResult with minimal metadata.
func loadBenchResult(path string) (*benchResult, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	data = bytes.TrimSpace(data)
	if len(data) == 0 {
		return nil, fmt.Errorf("empty file")
	}

	// v2: starts with '{'
	if data[0] == '{' {
		var result benchResult
		if err := json.Unmarshal(data, &result); err != nil {
			return nil, err
		}
		return &result, nil
	}

	// v1: starts with '[' — array of savedSeries
	var saved []savedSeries
	if err := json.Unmarshal(data, &saved); err != nil {
		return nil, err
	}

	result := &benchResult{Version: 1}
	for _, s := range saved {
		rs := richSeries{Name: s.Name}
		for _, p := range s.Points {
			rs.Points = append(rs.Points, richPoint{BPK: p.X, FPR: p.Y})
		}
		if s.Params != nil {
			var params map[string]interface{}
			if json.Unmarshal(s.Params, &params) == nil {
				if t, ok := params["type"].(string); ok {
					rs.FilterFamily = t
				}
			}
		}
		result.Series = append(result.Series, rs)
	}
	return result, nil
}

// richSeriesToPlotSeries extracts (bpk, fpr) pairs for SVG plotting.
func richSeriesToPlotSeries(rs richSeries, color, marker string, dashed bool) testutils.SeriesData {
	sd := testutils.SeriesData{
		Name:   rs.Name,
		Color:  color,
		Marker: marker,
		Dashed: dashed,
	}
	for _, p := range rs.Points {
		sd.Points = append(sd.Points, testutils.Point{X: p.BPK, Y: p.FPR})
	}
	return sd
}

// shouldSkipSeriesV2 decides whether a series should be skipped based on the
// v2 bench result's metadata. It checks ONLY/SKIP env vars and compares
// the existing series metadata hash.
func shouldSkipSeriesV2(name string, onlySet, skipSet map[string]bool, existing *benchResult) (bool, string) {
	if skipSet != nil && skipSet[name] {
		return true, "SKIP env var"
	}
	if onlySet != nil && !onlySet[name] {
		return true, "ONLY env var"
	}
	if existing != nil && existing.Version == 2 {
		for _, s := range existing.Series {
			if s.Name == name && len(s.Points) > 0 {
				return true, fmt.Sprintf("v2 cached, %d points", len(s.Points))
			}
		}
	}
	return false, ""
}

// v2CachedSeriesToPlotMap converts a v2 benchResult into the allSeries map format
// used by plotting, restoring cached points.
func v2CachedSeriesToPlotMap(existing *benchResult, allSeries map[string]*testutils.SeriesData) {
	if existing == nil {
		return
	}
	for _, rs := range existing.Series {
		if sd, ok := allSeries[rs.Name]; ok {
			sd.Points = nil
			for _, p := range rs.Points {
				sd.Points = append(sd.Points, testutils.Point{X: p.BPK, Y: p.FPR})
			}
		}
	}
}

// v2FindSeries returns the richSeries with the given name from a benchResult, or nil.
func v2FindSeries(result *benchResult, name string) *richSeries {
	if result == nil {
		return nil
	}
	for i := range result.Series {
		if result.Series[i].Name == name {
			return &result.Series[i]
		}
	}
	return nil
}

// v2SetSeries replaces or appends a richSeries in the benchResult.
func v2SetSeries(result *benchResult, rs richSeries) {
	for i := range result.Series {
		if result.Series[i].Name == rs.Name {
			result.Series[i] = rs
			return
		}
	}
	result.Series = append(result.Series, rs)
}

// newBenchMeta creates a benchMeta with current timestamp and git commit.
func newBenchMeta(benchType, distribution string, nKeys int, rangeLen uint64) benchMeta {
	return benchMeta{
		Type:         benchType,
		Distribution: distribution,
		NKeys:        nKeys,
		RangeLen:     rangeLen,
		Timestamp:    time.Now().UTC().Format(time.RFC3339),
		GitCommit:    gitCommitShort(),
	}
}

// newKeysMeta builds keysMeta from a benchConfig and pre-computed sha256.
func newKeysMeta(cfg benchConfig, keySHA string) keysMeta {
	return keysMeta{
		Source:           cfg.keySource,
		File:             cfg.keyFile,
		Seed:             cfg.keySeed,
		GenerationParams: cfg.keyGenParams,
		Count:            len(cfg.keys),
		SHA256:           keySHA,
	}
}

// newQueriesMeta builds queriesMeta from a benchConfig.
func newQueriesMeta(cfg benchConfig, seeds []int64, nRuns int) queriesMeta {
	return queriesMeta{
		Strategy:       cfg.queryStrategy,
		StrategyParams: cfg.queryStrategyParams,
		Count:          cfg.queryCount,
		Seeds:          seeds,
		NRuns:          nRuns,
	}
}
