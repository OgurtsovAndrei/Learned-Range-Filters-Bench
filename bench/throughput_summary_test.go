package bench_test

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// TestBuildThroughputSummary aggregates per-distribution build_throughput JSONs
// into a single cross-distribution table (filter × distribution at N=2^24)
// and writes a Markdown report plus a CSV. Runs only in PLOT_ONLY mode.
//
// Goal: visually answer two questions in one place —
//  1. Is build throughput nearly distribution-invariant for each filter?
//  2. Is per-key build time stable across N (linear scaling)?
func TestBuildThroughputSummary(t *testing.T) {
	if os.Getenv("PLOT_ONLY") == "" && os.Getenv("SUMMARIZE") == "" {
		t.Skip("set SUMMARIZE=1 (or PLOT_ONLY=1) to run the throughput summary aggregator")
	}

	dataDir := "../bench_results/data/build_throughput"
	entries, err := os.ReadDir(dataDir)
	if err != nil {
		t.Fatalf("read %s: %v", dataDir, err)
	}

	type point struct {
		N         int64
		MKeysSec  float64
		BuildNs   int64
	}
	// filter -> distribution -> sweep (sorted by N)
	table := map[string]map[string][]point{}
	dists := map[string]struct{}{}
	filters := map[string]struct{}{}

	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		path := filepath.Join(dataDir, e.Name())
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Errorf("read %s: %v", path, err)
			continue
		}
		var doc struct {
			Benchmark struct {
				Distribution string `json:"distribution"`
				NKeys        int64  `json:"nKeys"`
				Timestamp    string `json:"timestamp"`
				GitCommit    string `json:"gitCommit"`
			} `json:"benchmark"`
			Series []struct {
				Name   string `json:"name"`
				Points []struct {
					SweepParam  float64 `json:"sweepParam"`
					BPK         float64 `json:"bpk"` // throughput M keys/sec (legacy field reuse)
					BuildTimeNs *int64  `json:"buildTimeNs"`
				} `json:"points"`
			} `json:"series"`
		}
		if err := json.Unmarshal(raw, &doc); err != nil {
			t.Errorf("parse %s: %v", path, err)
			continue
		}
		dist := doc.Benchmark.Distribution
		dists[dist] = struct{}{}
		for _, s := range doc.Series {
			filters[s.Name] = struct{}{}
			if _, ok := table[s.Name]; !ok {
				table[s.Name] = map[string][]point{}
			}
			pts := make([]point, 0, len(s.Points))
			for _, p := range s.Points {
				ns := int64(0)
				if p.BuildTimeNs != nil {
					ns = *p.BuildTimeNs
				}
				pts = append(pts, point{
					N:        int64(p.SweepParam),
					MKeysSec: p.BPK,
					BuildNs:  ns,
				})
			}
			sort.Slice(pts, func(i, j int) bool { return pts[i].N < pts[j].N })
			table[s.Name][dist] = pts
		}
	}

	if len(dists) == 0 {
		t.Fatalf("no build_throughput JSONs found in %s", dataDir)
	}

	// Stable orderings.
	distOrder := []string{
		"clustered", "uniform", "spread",
		"sosd_books", "sosd_fb", "sosd_wiki", "sosd_osm",
	}
	finalDists := []string{}
	for _, d := range distOrder {
		if _, ok := dists[d]; ok {
			finalDists = append(finalDists, d)
		}
	}
	for d := range dists {
		found := false
		for _, fd := range finalDists {
			if fd == d {
				found = true
				break
			}
		}
		if !found {
			finalDists = append(finalDists, d)
		}
	}

	// Filter order: native ARE family first, then industry baselines.
	filterOrder := []string{
		"Adaptive(t=0)", "SODA", "Truncation", "Hybrid",
		"Scan-ARE", "Greedy+Merge", "CDF-ARE", "BloomARE",
		"Grafite", "SNARF", "SuRF", "SuRFHash(8)", "SuRFReal(8)",
	}
	finalFilters := []string{}
	for _, f := range filterOrder {
		if _, ok := filters[f]; ok {
			finalFilters = append(finalFilters, f)
		}
	}
	for f := range filters {
		found := false
		for _, ff := range finalFilters {
			if ff == f {
				found = true
				break
			}
		}
		if !found {
			finalFilters = append(finalFilters, f)
		}
	}

	const headlineN = int64(1 << 24)

	outDir := "../bench_results/data/build_throughput"
	mdPath := filepath.Join(outDir, "summary.md")
	csvPath := filepath.Join(outDir, "summary.csv")

	mdBuf := &strings.Builder{}
	csvBuf := &strings.Builder{}

	fmt.Fprintf(mdBuf, "# Build Throughput — cross-distribution summary\n\n")
	fmt.Fprintf(mdBuf, "Throughput at **N = 2^24 = %d keys** in **M keys/sec**, ε=0.01, L=100.\n\n", headlineN)
	fmt.Fprintf(mdBuf, "Source data: `bench_results/data/build_throughput/<dist>.json`.\n\n")

	// === Headline table: filter × distribution at N=2^24 ===
	mdBuf.WriteString("## Throughput @ N=2^24 (M keys/sec)\n\n")
	mdBuf.WriteString("| Filter |")
	csvBuf.WriteString("filter")
	for _, d := range finalDists {
		fmt.Fprintf(mdBuf, " %s |", d)
		fmt.Fprintf(csvBuf, ",%s", d)
	}
	mdBuf.WriteString(" min | max | max/min |\n")
	csvBuf.WriteString(",min,max,max_over_min\n")
	mdBuf.WriteString("|---|")
	for range finalDists {
		mdBuf.WriteString("---|")
	}
	mdBuf.WriteString("---|---|---|\n")

	for _, f := range finalFilters {
		fmt.Fprintf(mdBuf, "| %s |", f)
		fmt.Fprintf(csvBuf, "%s", f)
		var vmin, vmax = -1.0, -1.0
		for _, d := range finalDists {
			pts := table[f][d]
			val := -1.0
			for _, p := range pts {
				if p.N == headlineN {
					val = p.MKeysSec
					break
				}
			}
			if val < 0 {
				mdBuf.WriteString(" — |")
				csvBuf.WriteString(",")
				continue
			}
			fmt.Fprintf(mdBuf, " %.1f |", val)
			fmt.Fprintf(csvBuf, ",%.3f", val)
			if vmin < 0 || val < vmin {
				vmin = val
			}
			if val > vmax {
				vmax = val
			}
		}
		if vmin > 0 {
			fmt.Fprintf(mdBuf, " %.1f | %.1f | %.2f× |\n", vmin, vmax, vmax/vmin)
			fmt.Fprintf(csvBuf, ",%.3f,%.3f,%.3f\n", vmin, vmax, vmax/vmin)
		} else {
			mdBuf.WriteString(" — | — | — |\n")
			csvBuf.WriteString(",,,\n")
		}
	}

	// === Linearity check: ns/key vs N for `uniform` (or first available distribution). ===
	mdBuf.WriteString("\n## Linearity check — ns per key vs N\n\n")
	mdBuf.WriteString("If build is linear in N, ns/key (= buildTimeNs / N) should be roughly constant across N.\n")
	mdBuf.WriteString("A clear upward trend indicates super-linear scaling.\n\n")

	for _, dist := range finalDists {
		mdBuf.WriteString("### " + dist + " — ns/key vs N\n\n")
		mdBuf.WriteString("| Filter |")
		// Collect all N values present for this distribution.
		nset := map[int64]struct{}{}
		for _, f := range finalFilters {
			for _, p := range table[f][dist] {
				nset[p.N] = struct{}{}
			}
		}
		ns := []int64{}
		for n := range nset {
			ns = append(ns, n)
		}
		sort.Slice(ns, func(i, j int) bool { return ns[i] < ns[j] })
		for _, n := range ns {
			if n >= 1<<20 {
				fmt.Fprintf(mdBuf, " N=2^%d |", log2int(n))
			} else {
				fmt.Fprintf(mdBuf, " N=%d |", n)
			}
		}
		mdBuf.WriteString("\n|---|")
		for range ns {
			mdBuf.WriteString("---|")
		}
		mdBuf.WriteString("\n")

		for _, f := range finalFilters {
			pts := table[f][dist]
			if len(pts) == 0 {
				continue
			}
			fmt.Fprintf(mdBuf, "| %s |", f)
			byN := map[int64]point{}
			for _, p := range pts {
				byN[p.N] = p
			}
			for _, n := range ns {
				p, ok := byN[n]
				if !ok || p.BuildNs == 0 {
					mdBuf.WriteString(" — |")
					continue
				}
				nsPerKey := float64(p.BuildNs) / float64(n)
				fmt.Fprintf(mdBuf, " %.1f |", nsPerKey)
			}
			mdBuf.WriteString("\n")
		}
		mdBuf.WriteString("\n")
	}

	if err := os.WriteFile(mdPath, []byte(mdBuf.String()), 0644); err != nil {
		t.Fatalf("write %s: %v", mdPath, err)
	}
	if err := os.WriteFile(csvPath, []byte(csvBuf.String()), 0644); err != nil {
		t.Fatalf("write %s: %v", csvPath, err)
	}
	t.Logf("wrote %s", mdPath)
	t.Logf("wrote %s", csvPath)
}

func log2int(n int64) int {
	k := 0
	for n > 1 {
		n >>= 1
		k++
	}
	return k
}
