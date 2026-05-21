//go:build heavy

package bench_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestB6CacheMigration verifies the one-shot legacy single-file → per-filter
// directory migration in b6Store. It writes a synthetic legacy file, opens
// a store, flushes, and asserts:
//   - per-filter files exist for every Filter in the legacy doc
//   - rows lacking SweepName or Filter are dropped (legacy schema cleanup)
//   - the legacy file is renamed to *.legacy (one-shot guarantee)
//   - _meta.json is written with run-level invariants
//
// The test uses an isolated CWD so it never touches a real bench_results
// tree shared with the running sweep in main.
func TestB6CacheMigration(t *testing.T) {
	tmpRoot := t.TempDir()
	// b6Store paths are relative to CWD ("../bench_results/..."), so we
	// chdir into <tmpRoot>/bench/ for the duration of the test.
	benchDir := filepath.Join(tmpRoot, "bench")
	if err := os.MkdirAll(benchDir, 0o755); err != nil {
		t.Fatalf("mkdir benchDir: %v", err)
	}
	dataDir := filepath.Join(tmpRoot, "bench_results", "data")
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		t.Fatalf("mkdir dataDir: %v", err)
	}
	prevCwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	if err := os.Chdir(benchDir); err != nil {
		t.Fatalf("chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(prevCwd) })

	const N = 1 << 20
	legacy := b6Doc{
		Type:          "b6_latency",
		NKeys:         N,
		QueryCount:    1 << 18,
		QueryStrategy: "smart_mix_guaranteed_empty",
		Eps:           0.01,
		Timestamp:     "2026-04-30T10:00:00Z",
		GitCommit:     "abc1234",
		Rows: []b6Row{
			{Distribution: "uniform", Filter: "Grafite", RangeLen: 1, SweepName: "bpk", SweepParam: 12, BPKUsed: 12.3, ParamsHash: "h1"},
			{Distribution: "uniform", Filter: "Grafite", RangeLen: 16, SweepName: "bpk", SweepParam: 12, BPKUsed: 12.3, ParamsHash: "h2"},
			{Distribution: "uniform", Filter: "SODA", RangeLen: 1, SweepName: "K", SweepParam: 14, BPKUsed: 14, ParamsHash: "h3"},
			{Distribution: "sosd_fb", Filter: "SODA", RangeLen: 16, SweepName: "K", SweepParam: 14, BPKUsed: 14, ParamsHash: "h4"},
			// Drop expected: missing Filter.
			{Distribution: "uniform", Filter: "", RangeLen: 1, SweepName: "K", SweepParam: 0, ParamsHash: ""},
			// Drop expected: missing SweepName (legacy pre-K-sweep schema).
			{Distribution: "uniform", Filter: "Legacy_NoSweep", RangeLen: 1, SweepName: "", ParamsHash: ""},
		},
	}
	legacyPath := filepath.Join(dataDir, "b6_latency_N1048576.json")
	buf, err := json.MarshalIndent(legacy, "", "  ")
	if err != nil {
		t.Fatalf("marshal legacy: %v", err)
	}
	if err := os.WriteFile(legacyPath, buf, 0o644); err != nil {
		t.Fatalf("write legacy: %v", err)
	}

	store := newB6Store(N, 1<<18, 0.01, "smart_mix_guaranteed_empty", "")
	// Touching any filter triggers the one-shot migration.
	if doc := store.cachedRow("uniform", "Grafite", 1, "bpk", 12, "h1"); doc == nil {
		t.Fatalf("expected migrated Grafite row for uniform/L=1/bpk=12 to be present")
	}
	if err := store.flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}

	dirEntries, err := os.ReadDir(filepath.Join(dataDir, "b6_latency_N1048576"))
	if err != nil {
		t.Fatalf("read per-N dir: %v", err)
	}
	got := map[string]bool{}
	for _, e := range dirEntries {
		got[e.Name()] = true
	}
	for _, want := range []string{"_meta.json", "Grafite.json", "SODA.json"} {
		if !got[want] {
			t.Errorf("missing %s in per-N dir; got %v", want, got)
		}
	}
	if got["Legacy_NoSweep.json"] {
		t.Errorf("legacy SweepName-less filter should not have been migrated")
	}
	if got[".json"] {
		t.Errorf("empty-filter row should not have produced .json file")
	}

	// Legacy file must be renamed (.legacy suffix) so re-running won't
	// re-migrate.
	if _, err := os.Stat(legacyPath); !os.IsNotExist(err) {
		t.Errorf("legacy file still present at %s — migration did not rename", legacyPath)
	}
	if _, err := os.Stat(legacyPath + ".legacy"); err != nil {
		t.Errorf("expected .legacy renamed file at %s: %v", legacyPath+".legacy", err)
	}

	// Per-filter doc shape sanity.
	gbuf, err := os.ReadFile(filepath.Join(dataDir, "b6_latency_N1048576", "Grafite.json"))
	if err != nil {
		t.Fatalf("read Grafite.json: %v", err)
	}
	var gdoc b6FilterDoc
	if err := json.Unmarshal(gbuf, &gdoc); err != nil {
		t.Fatalf("parse Grafite.json: %v", err)
	}
	if gdoc.Type != "b6_latency_filter" {
		t.Errorf("Grafite.json type = %q, want b6_latency_filter", gdoc.Type)
	}
	if gdoc.Filter != "Grafite" || gdoc.NKeys != N || gdoc.SchemaVersion != b6SchemaVersion {
		t.Errorf("Grafite.json header mismatch: %+v", gdoc)
	}
	if len(gdoc.Rows) != 2 {
		t.Errorf("Grafite.json row count = %d, want 2", len(gdoc.Rows))
	}

	// _meta.json sanity.
	mbuf, err := os.ReadFile(filepath.Join(dataDir, "b6_latency_N1048576", "_meta.json"))
	if err != nil {
		t.Fatalf("read _meta.json: %v", err)
	}
	var meta b6MetaDoc
	if err := json.Unmarshal(mbuf, &meta); err != nil {
		t.Fatalf("parse _meta.json: %v", err)
	}
	if meta.NKeys != N || meta.Type != "b6_latency" || meta.SchemaVersion != b6SchemaVersion {
		t.Errorf("_meta.json header mismatch: %+v", meta)
	}

	// Re-opening the store must read the per-filter files cleanly and
	// must NOT re-migrate (legacy file is gone).
	store2 := newB6Store(N, 1<<18, 0.01, "smart_mix_guaranteed_empty", "")
	if cached := store2.cachedRow("uniform", "Grafite", 16, "bpk", 12, "h2"); cached == nil {
		t.Errorf("re-opened store lost migrated Grafite row for L=16")
	}
	if cached := store2.cachedRow("sosd_fb", "SODA", 16, "K", 14, "h4"); cached == nil {
		t.Errorf("re-opened store lost migrated SODA row for sosd_fb/L=16")
	}
}

// TestB6PlotSourceDiscovery exercises discoverB6PlotSources +
// loadB6PlotSource end-to-end. We seed a tempdir with both a per-filter
// directory and a legacy single-file (different N), assert they are both
// discovered, and assert the directory shadows a same-N legacy file.
func TestB6PlotSourceDiscovery(t *testing.T) {
	dataDir := t.TempDir()

	// Per-N directory format.
	dirN := filepath.Join(dataDir, "b6_latency_N16384")
	if err := os.MkdirAll(dirN, 0o755); err != nil {
		t.Fatalf("mkdir dirN: %v", err)
	}
	meta := b6MetaDoc{
		Type: "b6_latency", NKeys: 16384, QueryCount: 100,
		QueryStrategy: "smart_mix_guaranteed_empty", Eps: 0.01,
		SchemaVersion: b6SchemaVersion, CreatedAt: "2026-05-01T00:00:00Z",
	}
	mb, _ := json.Marshal(meta)
	if err := os.WriteFile(filepath.Join(dirN, "_meta.json"), mb, 0o644); err != nil {
		t.Fatalf("write meta: %v", err)
	}
	fdoc := b6FilterDoc{
		Type: "b6_latency_filter", Filter: "SODA", NKeys: 16384, QueryCount: 100,
		QueryStrategy: "smart_mix_guaranteed_empty", Eps: 0.01, SchemaVersion: b6SchemaVersion,
		Rows: []b6Row{
			{Distribution: "uniform", Filter: "SODA", RangeLen: 1, SweepName: "K", SweepParam: 14, ParamsHash: "h1"},
		},
	}
	fb, _ := json.Marshal(fdoc)
	if err := os.WriteFile(filepath.Join(dirN, "SODA.json"), fb, 0o644); err != nil {
		t.Fatalf("write filter: %v", err)
	}

	// Legacy single-file with a different N (must be picked up as fallback).
	legacy := b6Doc{
		Type: "b6_latency", NKeys: 32768, QueryCount: 200, QueryStrategy: "smart_mix_guaranteed_empty",
		Eps: 0.01,
		Rows: []b6Row{
			{Distribution: "uniform", Filter: "Grafite", RangeLen: 1, SweepName: "bpk", SweepParam: 12, ParamsHash: "h2"},
		},
	}
	lb, _ := json.Marshal(legacy)
	if err := os.WriteFile(filepath.Join(dataDir, "b6_latency_N32768.json"), lb, 0o644); err != nil {
		t.Fatalf("write legacy: %v", err)
	}

	// Same-N legacy that MUST be shadowed by the directory above.
	if err := os.WriteFile(filepath.Join(dataDir, "b6_latency_N16384.json"), lb, 0o644); err != nil {
		t.Fatalf("write shadowed: %v", err)
	}

	srcs, err := discoverB6PlotSources(dataDir)
	if err != nil {
		t.Fatalf("discover: %v", err)
	}
	if len(srcs) != 2 {
		t.Fatalf("expected 2 sources (1 dir + 1 legacy), got %d: %+v", len(srcs), srcs)
	}
	// First source should be the directory (sort order: dirs first).
	if srcs[0].dir == "" {
		t.Errorf("expected first source to be directory, got %+v", srcs[0])
	}
	if srcs[1].file == "" || srcs[1].label != "b6_latency_N32768.json" {
		t.Errorf("expected second source to be legacy file b6_latency_N32768.json, got %+v", srcs[1])
	}

	doc, err := loadB6PlotSource(srcs[0])
	if err != nil {
		t.Fatalf("load dir source: %v", err)
	}
	if doc.NKeys != 16384 || len(doc.Rows) != 1 || doc.Rows[0].Filter != "SODA" {
		t.Errorf("dir source content mismatch: %+v", doc)
	}
}

// TestB6CacheUpdateDedup verifies that update() replaces rows by
// paramsHash within the same filter doc, and that rows for a different
// paramsHash (e.g. a different parallelism) are preserved.
func TestB6CacheUpdateDedup(t *testing.T) {
	tmpRoot := t.TempDir()
	benchDir := filepath.Join(tmpRoot, "bench")
	if err := os.MkdirAll(benchDir, 0o755); err != nil {
		t.Fatalf("mkdir benchDir: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(tmpRoot, "bench_results", "data"), 0o755); err != nil {
		t.Fatalf("mkdir dataDir: %v", err)
	}
	prevCwd, _ := os.Getwd()
	if err := os.Chdir(benchDir); err != nil {
		t.Fatalf("chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(prevCwd) })

	store := newB6Store(1<<20, 1<<18, 0.01, "smart_mix_guaranteed_empty", "")

	// Initial rows at parallelism=1 and parallelism=4 (different paramsHash).
	store.update("uniform", "SODA", []b6Row{
		{Distribution: "uniform", Filter: "SODA", RangeLen: 1, SweepName: "K", SweepParam: 14, Parallelism: 1, ParamsHash: "p1L1"},
		{Distribution: "uniform", Filter: "SODA", RangeLen: 1, SweepName: "K", SweepParam: 14, Parallelism: 4, ParamsHash: "p4L1"},
	})

	// Re-update only the parallelism=1 row. The parallelism=4 row must be preserved.
	store.update("uniform", "SODA", []b6Row{
		{Distribution: "uniform", Filter: "SODA", RangeLen: 1, SweepName: "K", SweepParam: 14, Parallelism: 1, ParamsHash: "p1L1", QueryNsPerOp: 999},
	})

	r := store.cachedRow("uniform", "SODA", 1, "K", 14, "p1L1")
	if r == nil || r.QueryNsPerOp != 999 {
		t.Errorf("expected updated p1L1 row with QueryNsPerOp=999, got %+v", r)
	}
	if r := store.cachedRow("uniform", "SODA", 1, "K", 14, "p4L1"); r == nil {
		t.Errorf("p4L1 row was lost — different-paramsHash rows must be preserved")
	}
}
