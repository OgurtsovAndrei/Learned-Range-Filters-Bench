package bench_test

import (
	"Thesis/bits"
	"Thesis/emptiness/are_hybrid_scan"
	"Thesis/testutils"
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

// policyEntry pairs a display name and color with a FallbackPolicy.
type policyEntry struct {
	name   string
	color  string
	marker string
	policy func(epsilon float64) are_hybrid_scan.FallbackPolicy
}

var fallbackPolicies = []policyEntry{
	{
		name:   "Auto",
		color:  "#06b6d4",
		marker: "circle",
		policy: func(_ float64) are_hybrid_scan.FallbackPolicy { return are_hybrid_scan.FallbackAuto{} },
	},
	{
		name:   "AlwaysTrunc",
		color:  "#9b59b6",
		marker: "triangle",
		policy: func(_ float64) are_hybrid_scan.FallbackPolicy { return are_hybrid_scan.FallbackAlwaysTrunc{} },
	},
	{
		name:   "AlwaysSODA",
		color:  "#4dd88a",
		marker: "diamond",
		policy: func(_ float64) are_hybrid_scan.FallbackPolicy { return are_hybrid_scan.FallbackAlwaysSODA{} },
	},
	{
		name:   "EstFPR",
		color:  "#ff922b",
		marker: "star",
		policy: func(epsilon float64) are_hybrid_scan.FallbackPolicy {
			return are_hybrid_scan.FallbackEstimateFPR{Epsilon: epsilon}
		},
	},
	{
		name:   "GapFrac",
		color:  "#e74c3c",
		marker: "square",
		policy: func(epsilon float64) are_hybrid_scan.FallbackPolicy {
			return are_hybrid_scan.FallbackGapFraction{Epsilon: epsilon}
		},
	},
}

// runFallbackPolicyBench benchmarks all 4 FallbackPolicy variants for a single distribution.
func runFallbackPolicyBench(t *testing.T, distName string, keys []uint64, queryFunc func(rangeLen uint64, seed int64) [][2]uint64) {
	t.Helper()
	const (
		n          = 1 << 18
		queryCount = 1 << 18
		nRuns      = 3
	)
	rangeLens := []uint64{128, 1024}
	kGrid := []uint32{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 28, 32, 36, 40, 48}
	seeds := []int64{12345, 54321, 99999}

	keysBS := make([]bits.BitString, len(keys))
	for i, v := range keys {
		keysBS[i] = testutils.TrieBS(v)
	}

	dataDir := fmt.Sprintf("../bench_results/data/fallback_policy/%s", distName)
	plotDir := fmt.Sprintf("../bench_results/plots/fallback_policy/%s", distName)
	os.MkdirAll(dataDir, 0755)
	os.MkdirAll(plotDir, 0755)

	for _, rangeLen := range rangeLens {
		t.Run(fmt.Sprintf("L=%d", rangeLen), func(t *testing.T) {
			dataPath := fmt.Sprintf("%s/L%d.json", dataDir, rangeLen)

			// Load cached points.
			type savedPoint struct{ X, Y float64 }
			type savedSeries struct {
				Name   string        `json:"name"`
				Points []savedPoint  `json:"points"`
				Params json.RawMessage `json:"params,omitempty"`
			}
			loadCache := func() map[string][]savedPoint {
				data, err := os.ReadFile(dataPath)
				if err != nil {
					return nil
				}
				var saved []savedSeries
				if err := json.Unmarshal(data, &saved); err != nil {
					return nil
				}
				m := make(map[string][]savedPoint, len(saved))
				for _, s := range saved {
					m[s.Name] = s.Points
				}
				return m
			}
			cached := loadCache()

			seriesMap := make(map[string]*testutils.SeriesData)
			for _, pe := range fallbackPolicies {
				seriesMap[pe.name] = &testutils.SeriesData{
					Name:   pe.name,
					Color:  pe.color,
					Marker: pe.marker,
				}
				if pts, ok := cached[pe.name]; ok {
					for _, p := range pts {
						seriesMap[pe.name].Points = append(seriesMap[pe.name].Points, testutils.Point{X: p.X, Y: p.Y})
					}
				}
			}

			// Compute current params fingerprint.
			type paramsT struct {
				KGrid      []uint32 `json:"kGrid"`
				RangeLen   uint64   `json:"rangeLen"`
				NKeys      int      `json:"nKeys"`
				QueryCount int      `json:"queryCount"`
				Seeds      []int64  `json:"seeds"`
				NRuns      int      `json:"nRuns"`
			}
			currentParams, _ := json.Marshal(paramsT{kGrid, rangeLen, len(keys), queryCount, seeds, nRuns})

			// Check ONLY/SKIP env vars.
			onlySet := parseEnvSet("ONLY")
			skipSet := parseEnvSet("SKIP")

			fmt.Printf("\n=== FallbackPolicy — %s (n=%d, L=%d, %d runs) ===\n", distName, len(keys), rangeLen, nRuns)

			type fprTask struct {
				policy string
				bpk    float64
				isEmpty func(a, b uint64) bool
			}
			var tasks []fprTask
			rebuiltPolicies := make(map[string]bool)

			for _, pe := range fallbackPolicies {
				name := pe.name
				if len(onlySet) > 0 && !onlySet[name] {
					fmt.Printf("[SKIP]   %-16s (not in ONLY)\n", name)
					continue
				}
				if skipSet[name] {
					fmt.Printf("[SKIP]   %-16s (in SKIP)\n", name)
					continue
				}

				// Skip if params match cached.
				if cachedRaw, ok := func() (json.RawMessage, bool) {
					data, err := os.ReadFile(dataPath)
					if err != nil {
						return nil, false
					}
					var saved []savedSeries
					if err := json.Unmarshal(data, &saved); err != nil {
						return nil, false
					}
					for _, s := range saved {
						if s.Name == name {
							return s.Params, true
						}
					}
					return nil, false
				}(); ok && strings.EqualFold(string(cachedRaw), string(currentParams)) {
					fmt.Printf("[CACHED] %-16s\n", name)
					continue
				}

				fmt.Printf("[BUILD]  %-16s\n", name)
				seriesMap[name].Points = nil
				rebuiltPolicies[name] = true

				pe := pe
				for _, K := range kGrid {
					K := K
					// Back-compute epsilon for EstFPR policy.
					effectiveRL := float64(rangeLen) + 1
					epsilon := float64(len(keys)) * effectiveRL / math.Pow(2, float64(K))
					if epsilon <= 0 || epsilon > 1 {
						epsilon = 0.01
					}

					f, err := are_hybrid_scan.NewHybridScanAREWithPolicy(keysBS, rangeLen, K, pe.policy(epsilon))
					if err != nil {
						continue
					}
					bpk := float64(f.SizeInBits()) / float64(len(keys))
					f2 := f // capture
					tasks = append(tasks, fprTask{name, bpk, func(a, b uint64) bool {
						return f2.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b))
					}})
				}
			}

			// Run FPR measurements in parallel.
			type result struct {
				policy string
				point  testutils.Point
			}
			results := make([]result, len(tasks))
			var wg sync.WaitGroup
			for i, task := range tasks {
				i, task := i, task
				wg.Add(1)
				go func() {
					defer wg.Done()
					fpr := avgFPRParallel(keys, queryFunc, rangeLen, seeds, task.isEmpty)
					results[i] = result{task.policy, testutils.Point{X: task.bpk, Y: fpr}}
				}()
			}
			wg.Wait()

			for _, r := range results {
				seriesMap[r.policy].Points = append(seriesMap[r.policy].Points, r.point)
			}

			// Sort points by X for each series.
			for _, sd := range seriesMap {
				sort.Slice(sd.Points, func(i, j int) bool { return sd.Points[i].X < sd.Points[j].X })
			}

			// Print summary.
			fmt.Printf("%-16s | %8s | %14s\n", "Policy", "BPK", "FPR(avg)")
			fmt.Println(strings.Repeat("-", 45))
			for _, pe := range fallbackPolicies {
				for _, p := range seriesMap[pe.name].Points {
					fmt.Printf("%-16s | %8.2f | %14.6f\n", pe.name, p.X, p.Y)
				}
			}

			// Save JSON.
			saved := make([]savedSeries, 0, len(fallbackPolicies))
			for _, pe := range fallbackPolicies {
				sd := seriesMap[pe.name]
				pts := make([]savedPoint, len(sd.Points))
				for i, p := range sd.Points {
					pts[i] = savedPoint{p.X, p.Y}
				}
				var params json.RawMessage
				if rebuiltPolicies[pe.name] {
					params = currentParams
				} else if raw, ok := func() (json.RawMessage, bool) {
					data, err := os.ReadFile(dataPath)
					if err != nil {
						return nil, false
					}
					var ss []savedSeries
					if err := json.Unmarshal(data, &ss); err != nil {
						return nil, false
					}
					for _, s := range ss {
						if s.Name == pe.name {
							return s.Params, true
						}
					}
					return nil, false
				}(); ok {
					params = raw
				}
				saved = append(saved, savedSeries{pe.name, pts, params})
			}
			if b, err := json.MarshalIndent(saved, "", "  "); err == nil {
				os.WriteFile(dataPath, b, 0644)
			}

			// Generate SVG.
			seriesList := make([]testutils.SeriesData, 0, len(fallbackPolicies))
			for _, pe := range fallbackPolicies {
				seriesList = append(seriesList, *seriesMap[pe.name])
			}
			svgPath := fmt.Sprintf("%s/L%d.svg", plotDir, rangeLen)
			err := testutils.GenerateTradeoffSVG(
				fmt.Sprintf("FallbackPolicy — %s (60-bit keys, n=%d, L=%d)", distName, len(keys), rangeLen),
				"Bits per Key (BPK)",
				"False Positive Rate (FPR)",
				seriesList,
				svgPath,
			)
			if err != nil {
				t.Errorf("SVG failed: %v", err)
			} else {
				fmt.Printf("Plot written to %s\n", svgPath)
			}
		})
	}
}

// --- per-distribution test functions ---

func TestFallbackPolicy_Clustered(t *testing.T) {
	const (
		n          = 1 << 18
		nClusters  = 5
		unifFrac   = 0.15
		queryCount = 1 << 18
		cacheDir   = "../bench/synthetic_data"
	)
	keysPath := fmt.Sprintf("%s/clustered_%d.bin", cacheDir, n)
	metaPath := fmt.Sprintf("%s/clustered_%d_meta.json", cacheDir, n)
	os.MkdirAll(cacheDir, 0755)

	var keys []uint64
	var clusters []testutils.ClusterInfo

	if cachedKeys, err := loadSyntheticKeys(keysPath); err == nil {
		if metaBytes, err := os.ReadFile(metaPath); err == nil {
			var meta []clusterMeta
			if json.Unmarshal(metaBytes, &meta) == nil {
				clusters = make([]testutils.ClusterInfo, len(meta))
				for i, m := range meta {
					clusters[i] = testutils.ClusterInfo{Center: m.Center, Stddev: m.Stddev}
				}
				keys = cachedKeys
			}
		}
	}
	if keys == nil {
		rng := rand.New(rand.NewSource(99))
		raw, cls := testutils.GenerateClusterDistribution(n, nClusters, unifFrac, rng)
		keys = mask60Keys(raw)
		clusters = cls
		saveSyntheticKeys(keysPath, keys)
		meta := make([]clusterMeta, len(clusters))
		for i, c := range clusters {
			meta[i] = clusterMeta{Center: c.Center, Stddev: c.Stddev}
		}
		if b, err := json.MarshalIndent(meta, "", "  "); err == nil {
			os.WriteFile(metaPath, b, 0644)
		}
	}

	runFallbackPolicyBench(t, "clustered", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		qrng := rand.New(rand.NewSource(seed))
		return mask60Queries(testutils.GenerateClusterQueries(queryCount, clusters, unifFrac, rangeLen, qrng))
	})
}

func TestFallbackPolicy_Uniform(t *testing.T) {
	const (
		n          = 1 << 18
		queryCount = 1 << 18
	)
	keys := cacheOrGenerate("../bench/synthetic_data", "uniform", n, func() []uint64 {
		return generateUniformKeys(n, rand.New(rand.NewSource(42)))
	})
	runFallbackPolicyBench(t, "uniform", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		return generateUniformQueries(queryCount, rangeLen, rand.New(rand.NewSource(seed)))
	})
}

func TestFallbackPolicy_Spread(t *testing.T) {
	const (
		n          = 1 << 18
		queryCount = 1 << 18
	)
	keys := cacheOrGenerate("../bench/synthetic_data", "spread", n, func() []uint64 {
		return generateSpreadKeys(n)
	})
	runFallbackPolicyBench(t, "spread", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		return generateUniformQueries(queryCount, rangeLen, rand.New(rand.NewSource(seed)))
	})
}

func TestFallbackPolicy_Zipfian(t *testing.T) {
	const (
		n          = 1 << 18
		nPrefixes  = 100
		queryCount = 1 << 18
		cacheDir   = "../bench/synthetic_data"
	)
	keysPath := fmt.Sprintf("%s/zipfian_%d.bin", cacheDir, n)
	prefixesPath := fmt.Sprintf("%s/zipfian_%d_prefixes.bin", cacheDir, n)
	os.MkdirAll(cacheDir, 0755)

	var keys, prefixes []uint64
	if k, err := loadSyntheticKeys(keysPath); err == nil {
		if p, err := loadSyntheticKeys(prefixesPath); err == nil {
			keys, prefixes = k, p
		}
	}
	if keys == nil {
		rng := rand.New(rand.NewSource(77))
		keys, prefixes = generateZipfianKeys(n, nPrefixes, rng)
		saveSyntheticKeys(keysPath, keys)
		saveSyntheticKeys(prefixesPath, prefixes)
	}

	runFallbackPolicyBench(t, "zipfian", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		return generateZipfianQueries(queryCount, prefixes, rangeLen, rand.New(rand.NewSource(seed)))
	})
}

func TestFallbackPolicy_SOSD_Facebook(t *testing.T) {
	const queryCount = 1 << 18
	keys, err := loadSOSDUint64(sosdPath("fb_200M_uint64"), 1<<18)
	if err != nil {
		t.Skipf("SOSD data not available: %v (run bench/sosd_data/download.sh)", err)
	}
	keys = mask60Keys(keys)
	runFallbackPolicyBench(t, "sosd_fb", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
	})
}

func TestFallbackPolicy_SOSD_Wiki(t *testing.T) {
	const queryCount = 1 << 18
	keys, err := loadSOSDUint64(sosdPath("wiki_ts_200M_uint64"), 1<<18)
	if err != nil {
		t.Skipf("SOSD data not available: %v (run bench/sosd_data/download.sh)", err)
	}
	keys = mask60Keys(keys)
	runFallbackPolicyBench(t, "sosd_wiki", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
	})
}

func TestFallbackPolicy_SOSD_OSM(t *testing.T) {
	const queryCount = 1 << 18
	keys, err := loadSOSDUint64(sosdPath("osm_cellids_800M_uint64"), 1<<18)
	if err != nil {
		t.Skipf("SOSD data not available: %v (run bench/sosd_data/download.sh)", err)
	}
	keys = mask60Keys(keys)
	runFallbackPolicyBench(t, "sosd_osm", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
	})
}

func TestFallbackPolicy_SOSD_Books(t *testing.T) {
	const queryCount = 1 << 18
	keys, err := loadSOSDUint32(sosdPath("books_200M_uint32"), 1<<18)
	if err != nil {
		t.Skipf("SOSD data not available: %v (run bench/sosd_data/download.sh)", err)
	}
	keys = mask60Keys(keys)
	runFallbackPolicyBench(t, "sosd_books", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
	})
}
