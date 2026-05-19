// compare_latency_mix loads the per-filter b6 cache files for two query
// workloads — the 100%-empty baseline ("gap_heavy") and the no-truncation
// mixed-realistic variant ("gap_heavy_mixed") — and emits a side-by-side
// table of query latency at matching (distribution, filter, L, sweep) cells.
//
// Usage:
//
//	go run ./bench/cmd/compare_latency_mix -n 1048576
//	go run ./bench/cmd/compare_latency_mix -n 16777216 -filter Grafite -dist sosd_wiki
//	go run ./bench/cmd/compare_latency_mix -n 1048576 -best-only
//
// -best-only keeps only the row with the minimum ns/op per (dist, filter, L)
// pair on the empty side (closest to the evaluation-table operating point),
// which makes the apples-to-apples latency delta easy to read.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type b6Row struct {
	Distribution string  `json:"distribution"`
	Filter       string  `json:"filter"`
	RangeLen     uint64  `json:"rangeLen"`
	QueryNsPerOp float64 `json:"queryNsPerOp"`
	BPKUsed      float64 `json:"bpkUsed"`
	FPR          float64 `json:"fpr"`
	SweepName    string  `json:"sweepName"`
	SweepParam   float64 `json:"sweepParam"`
	Note         string  `json:"note,omitempty"`
}

type b6FilterDoc struct {
	Filter string  `json:"filter"`
	Rows   []b6Row `json:"rows"`
}

type cellKey struct {
	dist, filter string
	L            uint64
	sweepName    string
	sweepParam   float64
}

func loadDir(dir string) (map[cellKey]b6Row, error) {
	out := map[cellKey]b6Row{}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") || e.Name() == "_meta.json" {
			continue
		}
		path := filepath.Join(dir, e.Name())
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", path, err)
		}
		var doc b6FilterDoc
		if err := json.Unmarshal(b, &doc); err != nil {
			return nil, fmt.Errorf("decode %s: %w", path, err)
		}
		for _, r := range doc.Rows {
			if r.Note != "" {
				continue
			}
			k := cellKey{r.Distribution, r.Filter, r.RangeLen, r.SweepName, r.SweepParam}
			out[k] = r
		}
	}
	return out, nil
}

func main() {
	var (
		nKeys     = flag.Int("n", 1048576, "key count (N) — selects the b6_latency_N<N> dir pair")
		filterSel = flag.String("filter", "", "limit to this filter (comma-separated allowed)")
		distSel   = flag.String("dist", "", "limit to this distribution (comma-separated allowed)")
		baseRoot  = flag.String("base-root", "bench_results/data", "root of per-N cache dirs")
		bestOnly  = flag.Bool("best-only", false, "keep only the min-ns/op row per (dist, filter, L)")
	)
	flag.Parse()

	emptyDir := filepath.Join(*baseRoot, fmt.Sprintf("b6_latency_N%d_gap_heavy", *nKeys))
	mixedDir := filepath.Join(*baseRoot, fmt.Sprintf("b6_latency_N%d_gap_heavy_mixed", *nKeys))

	empty, err := loadDir(emptyDir)
	if err != nil && !errorIsNotExist(err) {
		fatal(err)
	}
	mixed, err := loadDir(mixedDir)
	if err != nil && !errorIsNotExist(err) {
		fatal(err)
	}
	if len(empty) == 0 {
		fatal(fmt.Errorf("no empty-side rows in %s", emptyDir))
	}
	if len(mixed) == 0 {
		fatal(fmt.Errorf("no mixed-side rows in %s", mixedDir))
	}

	filterAllow := parseSet(*filterSel)
	distAllow := parseSet(*distSel)

	if *bestOnly {
		empty = keepBestPerLCell(empty, filterAllow, distAllow)
	}

	keys := make([]cellKey, 0, len(empty))
	for k := range empty {
		if filterAllow != nil && !filterAllow[k.filter] {
			continue
		}
		if distAllow != nil && !distAllow[k.dist] {
			continue
		}
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := keys[i], keys[j]
		if a.dist != b.dist {
			return a.dist < b.dist
		}
		if a.filter != b.filter {
			return a.filter < b.filter
		}
		if a.L != b.L {
			return a.L < b.L
		}
		return a.sweepParam < b.sweepParam
	})

	fmt.Printf("# %s vs %s\n\n", emptyDir, mixedDir)
	fmt.Printf("%-12s | %-13s | %-5s | %-12s | %-10s | %-10s | %-8s | %-10s\n",
		"Distribution", "Filter", "L", "sweep", "ns/op empty", "ns/op mixed", "Δ ns/op", "Δ %")
	fmt.Printf("%s\n", strings.Repeat("-", 102))

	var hits, misses int
	for _, k := range keys {
		eRow := empty[k]
		mRow, ok := mixed[k]
		if !ok {
			misses++
			fmt.Printf("%-12s | %-13s | %-5d | %s=%-10.4g | %-10.1f | %-10s | %-8s | %-10s\n",
				k.dist, k.filter, k.L, k.sweepName, k.sweepParam, eRow.QueryNsPerOp, "—", "—", "(no mixed row)")
			continue
		}
		hits++
		delta := mRow.QueryNsPerOp - eRow.QueryNsPerOp
		var pct string
		if eRow.QueryNsPerOp > 0 {
			pct = fmt.Sprintf("%+.1f%%", 100*delta/eRow.QueryNsPerOp)
		} else {
			pct = "—"
		}
		fmt.Printf("%-12s | %-13s | %-5d | %s=%-10.4g | %-10.1f | %-10.1f | %+8.1f | %-10s\n",
			k.dist, k.filter, k.L, k.sweepName, k.sweepParam, eRow.QueryNsPerOp, mRow.QueryNsPerOp, delta, pct)
	}
	fmt.Printf("\nMatched %d cells (%d on empty side missing on mixed side)\n", hits, misses)
}

// keepBestPerLCell keeps only the row with the minimum ns/op per
// (distribution, filter, L) tuple — useful with -best-only to read the
// table without the full sweep noise.
func keepBestPerLCell(in map[cellKey]b6Row, filterAllow, distAllow map[string]bool) map[cellKey]b6Row {
	type lKey struct {
		dist, filter string
		L            uint64
	}
	best := map[lKey]cellKey{}
	bestNs := map[lKey]float64{}
	for k, r := range in {
		if filterAllow != nil && !filterAllow[k.filter] {
			continue
		}
		if distAllow != nil && !distAllow[k.dist] {
			continue
		}
		lk := lKey{k.dist, k.filter, k.L}
		if cur, ok := bestNs[lk]; !ok || r.QueryNsPerOp < cur {
			bestNs[lk] = r.QueryNsPerOp
			best[lk] = k
		}
	}
	out := map[cellKey]b6Row{}
	for _, k := range best {
		if r, ok := in[k]; ok && !math.IsNaN(r.QueryNsPerOp) {
			out[k] = r
		}
	}
	return out
}

func parseSet(csv string) map[string]bool {
	v := strings.TrimSpace(csv)
	if v == "" {
		return nil
	}
	out := map[string]bool{}
	for _, t := range strings.Split(v, ",") {
		t = strings.TrimSpace(t)
		if t != "" {
			out[t] = true
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}

func errorIsNotExist(err error) bool {
	return err != nil && (os.IsNotExist(err) || isFSErrNotExist(err))
}

func isFSErrNotExist(err error) bool {
	var pe *fs.PathError
	if asErr(err, &pe) {
		return os.IsNotExist(pe.Err)
	}
	return false
}

// asErr is errors.As inlined so we don't pull the package for a single use.
func asErr(err error, target any) bool {
	type wrapper interface{ Unwrap() error }
	for err != nil {
		switch t := target.(type) {
		case **fs.PathError:
			if pe, ok := err.(*fs.PathError); ok {
				*t = pe
				return true
			}
		}
		w, ok := err.(wrapper)
		if !ok {
			return false
		}
		err = w.Unwrap()
	}
	return false
}
