package bench_test

import (
	"Thesis-bench-industry/thirdparty/grafite"
	"Thesis-bench-industry/thirdparty/snarf"
	"Thesis-bench-industry/thirdparty/surf"
	"Thesis/emptiness/approx/are_bloom"
	"Thesis/emptiness/approx/are_greedy_scan"
	"Thesis/emptiness/approx/are_hybrid_scan"
	"Thesis/emptiness/approx/are_soda_hash"
	"Thesis/testutils"
	"encoding/json"
	"fmt"
	"math"
	mathbits "math/bits"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
)

// computeRefinedBPK returns the additional BPK values to measure for one CGo
// series after its initial DefaultBPKSweep is complete. One refinement pass.
//
// points: existing (BPK, FPR) measurements for the series, sorted by BPK ascending.
// floor:  the noise floor computed via DefaultYFloor (below which FPR can't be observed).
// xMax:   hard X-axis cap (DefaultXMax).
//
// Returns: BPK values to additionally measure; empty if the curve is already
// well-resolved or has reached the floor below xMax.
func computeRefinedBPK(points []testutils.Point, floor, xMax float64) []float64 {
	if len(points) < 2 {
		return nil
	}
	var extra []float64
	// Mid-point insertion for steep drops.
	for i := 0; i < len(points)-1; i++ {
		bpkDelta := points[i+1].X - points[i].X
		if bpkDelta < AdaptiveBPKGap {
			continue
		}
		// Skip if either FPR is already at or below floor (no signal in the gap).
		if points[i].Y <= floor || points[i+1].Y <= floor {
			continue
		}
		logDrop := math.Abs(math.Log10(points[i].Y) - math.Log10(points[i+1].Y))
		if logDrop < AdaptiveLogFPRDrop {
			continue
		}
		mid := (points[i].X + points[i+1].X) / 2
		extra = append(extra, mid)
	}
	// Tail extension: one probe past the last measurement.
	last := points[len(points)-1]
	if last.Y > floor && last.X+AdaptiveTailStep <= xMax {
		extra = append(extra, last.X+AdaptiveTailStep)
	}
	return extra
}

func runTradeoffBench(t *testing.T, cfg benchConfig) {
	nRuns := DefaultNRuns

	rangeLens := DefaultRangeLens
	kGrid := DefaultKGrid
	bpkSweep := DefaultBPKSweep
	epsilons := DefaultEpsilons

	keySHA := sha256Keys(cfg.keys)

	os.MkdirAll(BenchResultsPlotsDir(cfg.n, cfg.distName), 0755)

	// Parse ONLY/SKIP env vars once (shared across all range lengths).
	onlySet := parseEnvSet("ONLY")
	skipSet := parseEnvSet("SKIP")

	for _, rangeLen := range rangeLens {
		t.Run(fmt.Sprintf("L=%d", rangeLen), func(t *testing.T) {
			// ---- series map (constructed from DefaultSeriesStyles) ----
			allSeries := make(map[string]*testutils.SeriesData, len(DefaultSeriesStyles))
			for name, style := range DefaultSeriesStyles {
				allSeries[name] = &testutils.SeriesData{
					Name: style.Name, Color: style.Color, Marker: style.Marker, Dashed: style.Dashed,
				}
			}

			// v2 rich data tracking (parallel to allSeries for SVG).
			richData := make(map[string]*richSeries)
			for name := range allSeries {
				family := "kgrid"
				switch name {
				case "Theoretical":
					family = "theoretical"
				case "BloomARE":
					family = "epsilon"
				case "Grafite", "SNARF", "SuRFReal(8)":
					family = "bpksweep"
				}
				richData[name] = &richSeries{Name: name, FilterFamily: family}
			}

			// Actual key bit-width (max key value, ceil log2). Distribution-
			// specific: SOSD-Books n=2^20 ≈ 22 bits, SOSD-OSM ≈ 56 bits,
			// synthetic uniform/zipfian/temporal capped at 60 bits by their
			// generators. Reported in plot title and terminal print so the
			// reader knows the universe each filter operated on.
			keyBits := mathbits.Len64(cfg.keys[len(cfg.keys)-1])
			if keyBits == 0 {
				keyBits = 1
			}

			dataDir := BenchResultsDataDir(cfg.n, cfg.distName)
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
				seeds := DefaultSeeds

				// Load existing cache for per-series skip logic.
				cached := loadCachedSeries(dataPath)

				// Also load v2 for rich data preservation.
				existingV2, _ := loadBenchResult(dataPath)

				// Pre-compute current params for each series group.
				paramsKGrid := buildParamsKGrid(kGrid, rangeLen, len(cfg.keys), cfg.queryCount, seeds, nRuns)
				paramsEpsilon := buildParamsEpsilon(epsilons, rangeLen, len(cfg.keys), cfg.queryCount, seeds, nRuns)
				paramsBPKSweep := buildParamsBPKSweep(bpkSweep, rangeLen, len(cfg.keys), cfg.queryCount, seeds, nRuns)
				paramsTheoretical := buildParamsTheoretical(kGrid, rangeLen)

				// Determine per-series params mapping (used for saving).
				seriesParams := map[string]json.RawMessage{
					"Theoretical":  paramsTheoretical,
					"SODA":         paramsKGrid,
					"Scan-ARE":     paramsKGrid,
					"Greedy+Merge": paramsKGrid,
					"BloomARE":     paramsEpsilon,
					"Grafite":      paramsBPKSweep,
					"SNARF":        paramsBPKSweep,
					"SuRFReal(8)":  paramsBPKSweep,
				}

				// Store params in richData for v2 cache compatibility.
				for name, p := range seriesParams {
					if rd, ok := richData[name]; ok {
						rd.Params = p
					}
				}

				// newSeriesParams tracks which params to record for rebuilt series.
				newParams := make(map[string]json.RawMessage)

				// Restore cached points for all series upfront (will be overwritten if rebuilt).
				for name, cs := range cached {
					if sd, ok := allSeries[name]; ok {
						sd.Points = cs.Points
					}
				}
				// Restore v2 rich data from existing file.
				if existingV2 != nil {
					for _, rs := range existingV2.Series {
						if rd, ok := richData[rs.Name]; ok {
							rd.Points = rs.Points
							rd.SweepValues = rs.SweepValues
						}
					}
				}

				fmt.Printf("\n=== Industry Comparison — %s (%d-bit keys, n=%d, L=%d, %d runs) ===\n", cfg.distName, keyBits, len(cfg.keys), rangeLen, nRuns)
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
					richData["Theoretical"].Points = nil
					richData["Theoretical"].SweepValues = kGrid
					for _, K := range kGrid {
						thEps := float64(rangeLen) / math.Exp2(float64(K))
						if thEps > 0 && thEps <= 1 {
							allSeries["Theoretical"].Points = append(allSeries["Theoretical"].Points,
								testutils.Point{X: float64(K), Y: thEps})
							richData["Theoretical"].Points = append(richData["Theoretical"].Points,
								richPoint{SweepParam: float64(K), BPK: float64(K), FPR: thEps})
						}
					}
				}

				// ---- Build & measure ARE filters in parallel (pure Go, thread-safe) ----
				type fprTask struct {
					series         string
					label          string
					bpk            float64
					isEmpty        func(a, b uint64) bool
					sweepParam     float64
					filterSizeBits uint64
					filterStats    map[string]interface{}
				}
				// Determine which K-grid series to rebuild (logs [CACHED]/[BUILD] once per series).
				kgridSeriesNames := []string{"SODA", "Scan-ARE", "Greedy+Merge"}
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
						richData[name].Points = nil
						richData[name].SweepValues = kGrid
					}

					type richResult struct {
						seriesPoint
						rich richPoint
					}

					// Pre-generate queries once (avoid repeated generation per K×series×seed).
					pregenQueries := make(map[int64][][2]uint64, len(seeds))
					for _, seed := range seeds {
						pregenQueries[seed] = cfg.queryFunc(rangeLen, seed)
					}

					// Process one K at a time: build filters → measure FPR → release memory.
					for _, K := range kGrid {
						K := K
						var kTasks []fprTask

						if rebuildKGridSeries["SODA"] {
							if f, err := are_soda_hash.NewSodaAREFromK(cfg.keys, K, int64(rangeLen)); err == nil {
								sizeBits := f.SizeInBits()
								bpk := float64(sizeBits) / float64(len(cfg.keys))
								kTasks = append(kTasks, fprTask{"SODA", fmt.Sprintf("SODA(K=%d)", K), bpk,
									func(a, b uint64) bool { return f.IsEmpty(a, b) },
									float64(K), sizeBits, nil})
							}
						}
						if rebuildKGridSeries["Scan-ARE"] {
							if f, err := are_hybrid_scan.NewHybridScanARE(cfg.keys, uint32(keyBits), are_hybrid_scan.Config{K: K}); err == nil {
								sizeBits := f.SizeInBits()
								bpk := float64(sizeBits) / float64(len(cfg.keys))
								nc, nf, nt := f.Stats()
								stats := map[string]interface{}{
									"numClusters":  nc,
									"fallbackKeys": nf,
									"totalKeys":    nt,
								}
								kTasks = append(kTasks, fprTask{"Scan-ARE", fmt.Sprintf("Scan-ARE(K=%d)", K), bpk,
									f.IsEmpty,
									float64(K), sizeBits, stats})
							}
						}
						if rebuildKGridSeries["Greedy+Merge"] {
							if f, err := are_greedy_scan.NewGreedyScanARE(cfg.keys, uint32(keyBits), are_greedy_scan.Config{K: K}); err == nil {
								sizeBits := f.SizeInBits()
								bpk := float64(sizeBits) / float64(len(cfg.keys))
								nc, nf, nt := f.Stats()
								stats := map[string]interface{}{
									"numClusters":  nc,
									"fallbackKeys": nf,
									"totalKeys":    nt,
								}
								kTasks = append(kTasks, fprTask{"Greedy+Merge", fmt.Sprintf("Greedy+Merge(K=%d)", K), bpk,
									f.IsEmpty,
									float64(K), sizeBits, stats})
							}
						}

						// Measure FPR for this K's filters in parallel, then release them.
						kResults := make([]richResult, len(kTasks))
						var wg sync.WaitGroup
						for i, task := range kTasks {
							i, task := i, task
							wg.Add(1)
							go func() {
								defer wg.Done()
								fpr := avgFPRWithQueries(cfg.keys, pregenQueries, seeds, task.isEmpty)
								kResults[i] = richResult{
									seriesPoint{task.series, testutils.Point{X: task.bpk, Y: fpr}, task.label},
									richPoint{
										SweepParam:     task.sweepParam,
										BPK:            task.bpk,
										FPR:            fpr,
										FilterSizeBits: task.filterSizeBits,
										FilterStats:    task.filterStats,
									},
								}
							}()
						}
						wg.Wait()

						for _, rr := range kResults {
							allSeries[rr.series].Points = append(allSeries[rr.series].Points, rr.point)
							richData[rr.series].Points = append(richData[rr.series].Points, rr.rich)
							fmt.Printf("%-16s | %8.2f | %14.6f\n", rr.label, rr.point.X, rr.point.Y)
						}
						// Filters for this K are now unreferenced; GC can reclaim.
						runtime.GC()
					}
				}

				// ---- Epsilon-loop filters (BloomARE) ----
				rebuildEpsilonSeries := make(map[string]bool)
				for _, name := range []string{"BloomARE"} {
					if d := decideSkip(name, paramsEpsilon); !d.skip {
						rebuildEpsilonSeries[name] = true
					}
				}

				if len(rebuildEpsilonSeries) > 0 {
					for name := range rebuildEpsilonSeries {
						allSeries[name].Points = nil
						richData[name].Points = nil
						richData[name].SweepValues = epsilons
					}

					type fprTaskEps struct {
						fprTask
						eps float64
					}
					var epsilonTasks []fprTaskEps
					for _, eps := range epsilons {
						if rebuildEpsilonSeries["BloomARE"] && rangeLen < 1<<16 {
							if f, err := are_bloom.NewBloomARE(cfg.keys, rangeLen, eps); err == nil {
								sizeBits := f.SizeInBits()
								bpk := float64(sizeBits) / float64(len(cfg.keys))
								epsilonTasks = append(epsilonTasks, fprTaskEps{fprTask{"BloomARE", "BloomARE", bpk,
									func(a, b uint64) bool { return f.IsEmpty(a, b) },
									eps, sizeBits, nil}, eps})
							}
						}
					}

					type richResult struct {
						seriesPoint
						rich richPoint
					}
					epsilonResults := make([]richResult, len(epsilonTasks))
					var wg sync.WaitGroup
					for i, task := range epsilonTasks {
						i, task := i, task
						wg.Add(1)
						go func() {
							defer wg.Done()
							fpr := avgFPRParallel(cfg.keys, cfg.queryFunc, rangeLen, seeds, task.isEmpty)
							epsilonResults[i] = richResult{
								seriesPoint{task.series, testutils.Point{X: task.bpk, Y: fpr}, task.label},
								richPoint{
									SweepParam:     task.sweepParam,
									BPK:            task.bpk,
									FPR:            fpr,
									FilterSizeBits: task.filterSizeBits,
								},
							}
						}()
					}
					wg.Wait()

					for _, rr := range epsilonResults {
						allSeries[rr.series].Points = append(allSeries[rr.series].Points, rr.point)
						richData[rr.series].Points = append(richData[rr.series].Points, rr.rich)
						fmt.Printf("%-16s | %8.2f | %14.6f\n", rr.label, rr.point.X, rr.point.Y)
					}
				}

				// ---- CGo filters: build & measure sequentially (not thread-safe) ----
				// Determine which CGo series to rebuild.
				cgoSeries := []string{"Grafite", "SNARF", "SuRFReal(8)"}
				rebuildCGoSeries := make(map[string]bool)
				for _, name := range cgoSeries {
					if d := decideSkip(name, paramsBPKSweep); !d.skip {
						rebuildCGoSeries[name] = true
					}
				}

				if len(rebuildCGoSeries) > 0 {
					for name := range rebuildCGoSeries {
						allSeries[name].Points = nil
						richData[name].Points = nil
						richData[name].SweepValues = bpkSweep
					}

					for _, bpk := range bpkSweep {
						if rebuildCGoSeries["Grafite"] {
							if f := tryGrafite(cfg.keys, bpk); f != nil {
								sizeBits := f.SizeInBits()
								actualBPK := float64(sizeBits) / float64(len(cfg.keys))
								fpr := avgFPRBatch(cfg.keys, cfg.queryFunc, rangeLen, seeds, f.QueryBatch)
								allSeries["Grafite"].Points = append(allSeries["Grafite"].Points,
									testutils.Point{X: actualBPK, Y: fpr})
								richData["Grafite"].Points = append(richData["Grafite"].Points,
									richPoint{SweepParam: bpk, BPK: actualBPK, FPR: fpr, FilterSizeBits: sizeBits})
								fmt.Printf("%-16s | %8.2f | %14.6f\n", fmt.Sprintf("Grafite(bpk=%.0f)", bpk), actualBPK, fpr)
							}
						}

						if rebuildCGoSeries["SNARF"] {
							f := snarf.New(cfg.keys, bpk)
							sizeBits := f.SizeInBits()
							actualBPK := float64(sizeBits) / float64(len(cfg.keys))
							fpr := avgFPRBatch(cfg.keys, cfg.queryFunc, rangeLen, seeds, f.QueryBatch)
							allSeries["SNARF"].Points = append(allSeries["SNARF"].Points,
								testutils.Point{X: actualBPK, Y: fpr})
							richData["SNARF"].Points = append(richData["SNARF"].Points,
								richPoint{SweepParam: bpk, BPK: actualBPK, FPR: fpr, FilterSizeBits: sizeBits})
							fmt.Printf("%-16s | %8.2f | %14.6f\n", fmt.Sprintf("SNARF(bpk=%.0f)", bpk), actualBPK, fpr)
						}
					}

					// ---- Adaptive single-pass midpoint refinement (Grafite, SNARF) ----
					floor := DefaultYFloor(cfg.queryCount, nRuns)
					for _, name := range []string{"Grafite", "SNARF"} {
						if !rebuildCGoSeries[name] {
							continue
						}
						sort.Slice(allSeries[name].Points, func(i, j int) bool {
							return allSeries[name].Points[i].X < allSeries[name].Points[j].X
						})
						extraBPK := computeRefinedBPK(allSeries[name].Points, floor, DefaultXMax)
						for _, bpk := range extraBPK {
							switch name {
							case "Grafite":
								if f := tryGrafite(cfg.keys, bpk); f != nil {
									sizeBits := f.SizeInBits()
									actualBPK := float64(sizeBits) / float64(len(cfg.keys))
									fpr := avgFPRBatch(cfg.keys, cfg.queryFunc, rangeLen, seeds, f.QueryBatch)
									allSeries["Grafite"].Points = append(allSeries["Grafite"].Points,
										testutils.Point{X: actualBPK, Y: fpr})
									richData["Grafite"].Points = append(richData["Grafite"].Points,
										richPoint{SweepParam: bpk, BPK: actualBPK, FPR: fpr, FilterSizeBits: sizeBits})
									fmt.Printf("%-16s | %8.2f | %14.6f\n", fmt.Sprintf("Grafite(bpk=%.2f)*", bpk), actualBPK, fpr)
								}
							case "SNARF":
								f := snarf.New(cfg.keys, bpk)
								sizeBits := f.SizeInBits()
								actualBPK := float64(sizeBits) / float64(len(cfg.keys))
								fpr := avgFPRBatch(cfg.keys, cfg.queryFunc, rangeLen, seeds, f.QueryBatch)
								allSeries["SNARF"].Points = append(allSeries["SNARF"].Points,
									testutils.Point{X: actualBPK, Y: fpr})
								richData["SNARF"].Points = append(richData["SNARF"].Points,
									richPoint{SweepParam: bpk, BPK: actualBPK, FPR: fpr, FilterSizeBits: sizeBits})
								fmt.Printf("%-16s | %8.2f | %14.6f\n", fmt.Sprintf("SNARF(bpk=%.2f)*", bpk), actualBPK, fpr)
							}
						}
						// Re-sort by BPK so plots render correctly.
						sort.Slice(allSeries[name].Points, func(i, j int) bool {
							return allSeries[name].Points[i].X < allSeries[name].Points[j].X
						})
						sort.Slice(richData[name].Points, func(i, j int) bool {
							return richData[name].Points[i].BPK < richData[name].Points[j].BPK
						})
					}

					type surfVariant struct {
						name     string
						st       surf.SuffixType
						hashBits int
						realBits int
					}
					for _, sv := range []surfVariant{
						{"SuRFReal(8)", surf.SuffixReal, 0, 8},
					} {
						if rebuildCGoSeries[sv.name] {
							f := surf.New(cfg.keys, sv.st, sv.hashBits, sv.realBits)
							sizeBits := f.SizeInBits()
							actualBPK := float64(sizeBits) / float64(len(cfg.keys))
							fpr := avgFPRBatch(cfg.keys, cfg.queryFunc, rangeLen, seeds, f.QueryBatch)
							allSeries[sv.name].Points = append(allSeries[sv.name].Points,
								testutils.Point{X: actualBPK, Y: fpr})
							richData[sv.name].Points = append(richData[sv.name].Points,
								richPoint{SweepParam: 0, BPK: actualBPK, FPR: fpr, FilterSizeBits: sizeBits})
							fmt.Printf("%-16s | %8.2f | %14.6f\n", sv.name, actualBPK, fpr)
						}
					}
				}

				// ---- Save v2 benchResult ----
				v2Result := &benchResult{
					Version:   2,
					Benchmark: newBenchMeta("fpr_tradeoff", cfg.distName, len(cfg.keys), rangeLen),
					Keys:      newKeysMeta(cfg, keySHA),
					Queries:   newQueriesMeta(cfg, seeds, nRuns),
				}
				for _, name := range []string{
					"Theoretical", "Grafite", "SNARF", "SuRFReal(8)",
					"SODA", "Scan-ARE", "Greedy+Merge",
					"BloomARE",
				} {
					rs := *richData[name]
					// For series not rebuilt this run, preserve existing v2 data.
					if _, rebuilt := newParams[name]; !rebuilt {
						if existingV2 != nil {
							if es := v2FindSeries(existingV2, name); es != nil && len(es.Points) > 0 {
								rs = *es
							}
						}
					}
					if len(rs.Points) > 0 {
						v2Result.Series = append(v2Result.Series, rs)
					}
				}
				if err := saveBenchResult(dataPath, v2Result); err != nil {
					t.Logf("warning: failed to save v2 data: %v", err)
				} else {
					// Reload to ensure allSeries reflects the complete merged state.
					reloaded := loadCachedSeries(dataPath)
					for name, cs := range reloaded {
						if sd, ok := allSeries[name]; ok {
							sd.Points = cs.Points
						}
					}
				}

			}

			// ---- Annotate Grafite saturation if its library guard would clip
			// inside the plot range. The X-marker + caption signal that
			// Grafite cannot be measured beyond log2(U/n)+2 — not that we
			// stopped sweeping prematurely.
			//
			// Two cases:
			//   * Some sweep points succeed: the last measured point gets
			//     replaced by an X (last.X < maxBPK <= last.X + sweepStep).
			//   * No sweep point is below maxBPK (small-universe distros
			//     like Books): seed a single phantom point at (maxBPK, 1.0)
			//     so the X lands on the X-axis at exactly the library
			//     boundary. Without this, the series is empty and the
			//     plot silently omits Grafite.
			if g := allSeries["Grafite"]; g != nil && len(cfg.keys) >= 2 {
				universe := cfg.keys[len(cfg.keys)-1] - cfg.keys[0]
				if universe > 0 {
					maxBPK := math.Log2(float64(universe)/float64(len(cfg.keys))) + 2
					if maxBPK < DefaultXMax {
						if len(g.Points) == 0 {
							g.Points = []testutils.Point{{X: maxBPK, Y: 1.0}}
						}
						g.EndStop = true
						g.EndCaption = "(library limit)"
					}
				}
			}

			// ---- Generate SVG ----
			orderedSeries := []testutils.SeriesData{
				*allSeries["Theoretical"],
				*allSeries["Grafite"],
				*allSeries["SNARF"],
				*allSeries["SuRFReal(8)"],
				*allSeries["SODA"],
				*allSeries["Scan-ARE"],
				*allSeries["Greedy+Merge"],
				*allSeries["BloomARE"],
			}

			svgPath := fmt.Sprintf("%s/L%d.svg", BenchResultsPlotsDir(cfg.n, cfg.distName), rangeLen)
			err := testutils.GenerateTradeoffSVG(
				fmt.Sprintf("FPR vs BPK — %s (%d-bit keys, n=%d, L=%d)", cfg.distName, keyBits, len(cfg.keys), rangeLen),
				"Bits per Key (BPK)",
				"False Positive Rate (FPR)",
				orderedSeries,
				svgPath,
				DefaultYFloor(cfg.queryCount, nRuns),
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
				keys = rawKeys
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

			keySeed := int64(99)
			runTradeoffBench(t, benchConfig{
				distName:   "clustered",
				n:          n,
				keys:       keys,
				queryCount: queryCount,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					qrng := rand.New(rand.NewSource(seed))
					return testutils.GenerateClusterQueries(queryCount, clusters, unifFrac, rangeLen, qrng)
				},
				keySource:     "synthetic",
				keyFile:       fmt.Sprintf("clustered_%d.bin", n),
				keySeed:       &keySeed,
				keyGenParams:  map[string]interface{}{"nClusters": nClusters, "unifFrac": unifFrac},
				queryStrategy: "cluster",
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
			keySeed := int64(42)
			runTradeoffBench(t, benchConfig{
				distName:   "uniform",
				n:          n,
				keys:       keys,
				queryCount: queryCount,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					qrng := rand.New(rand.NewSource(seed))
					return generateUniformQueries(queryCount, rangeLen, qrng)
				},
				keySource:     "synthetic",
				keyFile:       fmt.Sprintf("uniform_%d.bin", n),
				keySeed:       &keySeed,
				queryStrategy: "uniform",
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
				keySource:     "synthetic",
				keyFile:       fmt.Sprintf("spread_%d.bin", n),
				queryStrategy: "uniform",
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

			keySeed := int64(77)
			runTradeoffBench(t, benchConfig{
				distName:   "zipfian",
				n:          n,
				keys:       keys,
				queryCount: queryCount,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					qrng := rand.New(rand.NewSource(seed))
					return generateZipfianQueries(queryCount, prefixes, rangeLen, qrng)
				},
				keySource:     "synthetic",
				keyFile:       fmt.Sprintf("zipfian_%d.bin", n),
				keySeed:       &keySeed,
				keyGenParams:  map[string]interface{}{"nPrefixes": nPrefixes},
				queryStrategy: "zipfian",
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
			keySeed := int64(55)
			runTradeoffBench(t, benchConfig{
				distName:   "temporal",
				n:          n,
				keys:       keys,
				queryCount: queryCount,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					qrng := rand.New(rand.NewSource(seed))
					return generateTemporalQueries(queryCount, keys, rangeLen, qrng)
				},
				keySource:     "synthetic",
				keyFile:       fmt.Sprintf("temporal_%d.bin", n),
				keySeed:       &keySeed,
				queryStrategy: "temporal",
			})
		})
	}
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
			return raw
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
		XMax:   DefaultXMax,
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
		XMax:   DefaultXMax,
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
			XMax:   DefaultXMax,
		}, histSeries, path)
		if err != nil {
			t.Errorf("histogram SVG failed for %s: %v", d.name, err)
		} else {
			fmt.Printf("Histogram written to %s\n", path)
		}
	}
}
