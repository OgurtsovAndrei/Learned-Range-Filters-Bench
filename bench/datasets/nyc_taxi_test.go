//go:build heavy

package datasets_test

import (
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"Thesis-bench-industry/bench/datasets"
)

func nycTaxiPath(name string) string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), "..", "nyc_taxi_data", name)
}

// TestNYCTaxiPickupReader_OneFile picks any single parquet file in the
// nyc_taxi_data directory, reads pickup timestamps via the parquet reader,
// and asserts basic invariants (non-zero rows, sorted-unique, plausible
// timestamp range 2008..2027).
func TestNYCTaxiPickupReader_OneFile(t *testing.T) {
	matches, _ := filepath.Glob(nycTaxiPath("*_tripdata_*.parquet"))
	if len(matches) == 0 {
		t.Skip("no nyc_taxi_data parquet files (run download.sh)")
	}
	r := &datasets.NYCTaxiPickupReader{
		Files: matches[:1],
		Label: "nyc_taxi_smoke",
	}
	keys, err := r.Keys()
	if err != nil {
		t.Fatalf("Keys: %v", err)
	}
	if len(keys) == 0 {
		t.Fatalf("no keys read from %s", matches[0])
	}
	for i := 1; i < len(keys); i++ {
		if keys[i] <= keys[i-1] {
			t.Fatalf("not sorted-unique at i=%d: %d <= %d", i, keys[i], keys[i-1])
		}
	}
	// Plausible-range check: 2008-01-01 .. 2027-01-01 in Unix nanoseconds.
	lo := uint64(time.Date(2008, 1, 1, 0, 0, 0, 0, time.UTC).UnixNano())
	hi := uint64(time.Date(2027, 1, 1, 0, 0, 0, 0, time.UTC).UnixNano())
	if keys[0] < lo || keys[len(keys)-1] > hi {
		t.Errorf("timestamps out of range: min=%d max=%d (expected within %d..%d)",
			keys[0], keys[len(keys)-1], lo, hi)
	}
	t.Logf("file=%s n=%d range=[%s, %s]",
		filepath.Base(matches[0]),
		len(keys),
		time.Unix(0, int64(keys[0])).UTC().Format(time.RFC3339),
		time.Unix(0, int64(keys[len(keys)-1])).UTC().Format(time.RFC3339))
}

// TestNYCTaxiPickupReader_FromGlob exercises the glob constructor on the
// yellow_tripdata files, taking just the first 3 if many are present.
func TestNYCTaxiPickupReader_FromGlob(t *testing.T) {
	matches, _ := filepath.Glob(nycTaxiPath("yellow_tripdata_*.parquet"))
	if len(matches) == 0 {
		t.Skip("no yellow_tripdata parquet files")
	}
	if len(matches) > 3 {
		matches = matches[:3]
	}
	r := &datasets.NYCTaxiPickupReader{
		Files: matches,
		Label: "nyc_yellow_3",
	}
	keys, err := r.Keys()
	if err != nil {
		t.Fatalf("Keys: %v", err)
	}
	for i := 1; i < len(keys); i++ {
		if keys[i] <= keys[i-1] {
			t.Fatalf("multi-file output not sorted-unique at i=%d", i)
		}
	}
	t.Logf("files=%d n=%d", len(matches), len(keys))
}
