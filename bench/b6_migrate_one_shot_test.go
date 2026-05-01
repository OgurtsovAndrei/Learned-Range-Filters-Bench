package bench_test

import (
	"os"
	"strconv"
	"testing"
)

// TestB6MigrateLegacyOneShot is a manual-invocation utility that forces the
// legacy-to-per-filter migration for a given N without running any benchmark
// cells. Set B6_MIGRATE_N=<N> and run:
//
//	B6_MIGRATE_N=16777216 go test -run TestB6MigrateLegacyOneShot ./bench/
//
// Without B6_MIGRATE_N the test skips immediately. After migration the
// caller is free to surgically delete per-filter JSONs (e.g. CGo filters)
// before re-running TestB6IndustryLatency.
func TestB6MigrateLegacyOneShot(t *testing.T) {
	v := os.Getenv("B6_MIGRATE_N")
	if v == "" {
		t.Skip("set B6_MIGRATE_N=<N> to trigger legacy migration")
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 1 {
		t.Fatalf("B6_MIGRATE_N: bad value %q", v)
	}
	store := newB6Store(n, 1<<18, 0.01, "smart_mix_guaranteed_empty", "")
	// Touching any filter triggers migrateLegacyLocked via loadFilterLocked.
	_ = store.cachedRow("any", "Scan-ARE", 1, "K", 0, "_force_migration_")
	if err := store.flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}
	t.Logf("migration done; per-filter dir at %s", store.path())
}
