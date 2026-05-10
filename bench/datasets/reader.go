// Package datasets unifies access to benchmark key sources behind a single
// Reader interface. Implementations may load from SOSD binary files, NYC TLC
// parquet files, or any other format — bench code consumes []uint64 without
// caring about the on-disk format.
package datasets

// Reader yields benchmark keys as a sorted, deduplicated []uint64.
//
// Implementations are expected to be deterministic for a given input file
// set. Keys() may load lazily; callers should not assume the slice is cached.
type Reader interface {
	// Name returns a short, file-system-safe identifier used for plot
	// labels, result paths, and cache keys (e.g., "sosd_fb",
	// "nyc_yellow_pickup_2024").
	Name() string

	// Keys returns sorted unique keys. Returns an error if any backing
	// file is missing or malformed.
	Keys() ([]uint64, error)
}
