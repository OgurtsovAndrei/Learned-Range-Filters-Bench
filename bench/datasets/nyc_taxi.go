package datasets

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/parquet-go/parquet-go"
	"github.com/parquet-go/parquet-go/deprecated"
)

// NYCTaxiPickupReader reads pickup-timestamp keys directly from one or more
// NYC TLC trip-record parquet files. Timestamps are converted to Unix
// nanoseconds and returned as sorted-unique uint64 keys.
//
// Column-name detection: the reader probes each file's schema for the first
// matching candidate (yellow uses tpep_pickup_datetime, green uses
// lpep_pickup_datetime, fhv/fhvhv use pickup_datetime; pre-2015 yellow uses
// "Pickup_datetime" with various capitalizations).
type NYCTaxiPickupReader struct {
	// Files is the list of parquet files to read. May be one file or a
	// glob/dir result.
	Files []string

	// Label is the dataset name used for reporting.
	Label string

	// MaxKeys, if > 0, caps the returned slice (after sort+dedupe).
	MaxKeys int
}

// NYCTaxiPickupFromGlob constructs a reader that consumes every parquet file
// matching the glob (e.g., "bench/nyc_taxi_data/yellow_tripdata_*.parquet").
func NYCTaxiPickupFromGlob(label, glob string) (*NYCTaxiPickupReader, error) {
	matches, err := filepath.Glob(glob)
	if err != nil {
		return nil, fmt.Errorf("glob %q: %w", glob, err)
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("no parquet files match %q", glob)
	}
	sort.Strings(matches)
	return &NYCTaxiPickupReader{Files: matches, Label: label}, nil
}

func (r *NYCTaxiPickupReader) Name() string { return r.Label }

// RawTimestamps returns every pickup timestamp in the configured files,
// sorted but NOT deduplicated. Multiple trips occurring in the same second
// (or microsecond, depending on file precision) appear once per trip — this
// is what histograms of demand intensity want.
//
// If MaxKeys > 0, reading stops once that many timestamps have been
// accumulated (file order). Useful as a hard memory cap.
func (r *NYCTaxiPickupReader) RawTimestamps() ([]uint64, error) {
	return r.RawTimestampsInRange(0, 0)
}

// RawTimestampsInRange returns sorted, non-deduplicated pickup timestamps
// that fall in [lo, hi) Unix nanoseconds. lo == 0 means no lower bound;
// hi == 0 means no upper bound. The range filter is applied at the parquet
// reader so out-of-range values are never materialized.
func (r *NYCTaxiPickupReader) RawTimestampsInRange(lo, hi uint64) ([]uint64, error) {
	var all []uint64
	cap := r.MaxKeys
	for _, path := range r.Files {
		err := streamPickupColumn(path, lo, hi, func(ts uint64) bool {
			all = append(all, ts)
			return cap == 0 || len(all) < cap
		})
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", path, err)
		}
		if cap > 0 && len(all) >= cap {
			break
		}
	}
	sort.Slice(all, func(i, j int) bool { return all[i] < all[j] })
	return all, nil
}

// BinHistogram returns nBins counts of pickup timestamps falling in [lo, hi),
// each bin covering (hi-lo)/nBins ns. Memory is O(nBins) — values are
// streamed from parquet and accumulated directly into bin counters; the full
// timestamp list is never materialized. Suitable for year- or multi-year-
// scale aggregates.
func (r *NYCTaxiPickupReader) BinHistogram(lo, hi uint64, nBins int) ([]uint64, error) {
	if hi <= lo {
		return nil, fmt.Errorf("BinHistogram: invalid range [%d, %d)", lo, hi)
	}
	if nBins <= 0 {
		return nil, fmt.Errorf("BinHistogram: nBins must be > 0")
	}
	span := hi - lo
	// Compute bin index as (ts - lo) / binWidth to avoid uint64 overflow
	// when nBins * span exceeds 2^64 (e.g., year-scale with 1000 bins).
	binWidth := span / uint64(nBins)
	if binWidth == 0 {
		binWidth = 1
	}
	counts := make([]uint64, nBins)
	for _, path := range r.Files {
		err := streamPickupColumn(path, lo, hi, func(ts uint64) bool {
			bin := int((ts - lo) / binWidth)
			if bin >= nBins {
				bin = nBins - 1
			}
			counts[bin]++
			return true
		})
		if err != nil {
			return nil, fmt.Errorf("bin %s: %w", path, err)
		}
	}
	return counts, nil
}

func (r *NYCTaxiPickupReader) Keys() ([]uint64, error) {
	var all []uint64
	for _, path := range r.Files {
		ks, err := readPickupColumn(path)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", path, err)
		}
		all = append(all, ks...)
	}
	sort.Slice(all, func(i, j int) bool { return all[i] < all[j] })

	j := 0
	for i := 1; i < len(all); i++ {
		if all[i] != all[j] {
			j++
			all[j] = all[i]
		}
	}
	if len(all) > 0 {
		all = all[:j+1]
	}
	if r.MaxKeys > 0 && len(all) > r.MaxKeys {
		all = all[:r.MaxKeys]
	}
	return all, nil
}

// pickupColumnCandidates lists known pickup-timestamp column names across
// TLC service flavours. Match is case-insensitive.
var pickupColumnCandidates = []string{
	"tpep_pickup_datetime",
	"lpep_pickup_datetime",
	"pickup_datetime",
	"Pickup_datetime",
	"Trip_Pickup_DateTime",
}

// readPickupColumn is a thin wrapper around streamPickupColumn that
// accumulates all in-range timestamps into a slice. Used by Keys() where
// memory is bounded by the file's row count.
func readPickupColumn(path string) ([]uint64, error) {
	var keys []uint64
	err := streamPickupColumn(path, 0, 0, func(ts uint64) bool {
		keys = append(keys, ts)
		return true
	})
	return keys, err
}

// streamPickupColumn opens the parquet file at path, locates its pickup-
// timestamp column, and invokes yield(ts) for each value (in file order).
// If lo > 0 only timestamps >= lo are yielded; if hi > 0 only timestamps
// < hi are yielded. Reading stops early when yield returns false.
//
// Memory is O(read-buffer) — bin counters or downstream consumers should
// keep heap pressure flat regardless of file size.
func streamPickupColumn(path string, lo, hi uint64, yield func(ts uint64) bool) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return err
	}
	pf, err := parquet.OpenFile(f, st.Size())
	if err != nil {
		return fmt.Errorf("open parquet: %w", err)
	}

	colIdx := -1
	var colName string
	for i, fld := range pf.Schema().Fields() {
		for _, cand := range pickupColumnCandidates {
			if strings.EqualFold(fld.Name(), cand) {
				colIdx = i
				colName = fld.Name()
				break
			}
		}
		if colIdx >= 0 {
			break
		}
	}
	if colIdx < 0 {
		return errors.New("no pickup-timestamp column found")
	}

	colType := pf.Schema().Fields()[colIdx].Type()
	kind := colType.Kind()
	unitMul := timestampUnitMultiplier(colType)

	buf := make([]parquet.Value, 4096)

	for _, rg := range pf.RowGroups() {
		chunks := rg.ColumnChunks()
		if colIdx >= len(chunks) {
			return fmt.Errorf("column %s missing in row group", colName)
		}
		pages := chunks[colIdx].Pages()
		for {
			page, err := pages.ReadPage()
			if err != nil {
				if errors.Is(err, io.EOF) {
					break
				}
				pages.Close()
				return fmt.Errorf("read page: %w", err)
			}
			values := page.Values()
			for {
				n, err := values.ReadValues(buf)
				for k := 0; k < n; k++ {
					if buf[k].IsNull() {
						continue
					}
					ts, ok := valueToNanos(buf[k], kind, unitMul)
					if !ok {
						continue
					}
					if lo > 0 && ts < lo {
						continue
					}
					if hi > 0 && ts >= hi {
						continue
					}
					if !yield(ts) {
						parquet.Release(page)
						pages.Close()
						return nil
					}
				}
				if err != nil {
					if errors.Is(err, io.EOF) {
						break
					}
					parquet.Release(page)
					pages.Close()
					return fmt.Errorf("read values: %w", err)
				}
			}
			parquet.Release(page)
		}
		if err := pages.Close(); err != nil {
			return fmt.Errorf("close pages: %w", err)
		}
	}
	return nil
}

// valueToNanos converts a parquet Value to Unix nanoseconds based on its
// physical kind. INT64 values are scaled by unitMul (parquet timestamp unit);
// BYTE_ARRAY values are parsed as date strings ("YYYY-MM-DD HH:MM:SS" or
// "YYYY-MM-DDTHH:MM:SS"). Returns (ns, true) on success.
func valueToNanos(v parquet.Value, kind parquet.Kind, unitMul int64) (uint64, bool) {
	switch kind {
	case parquet.Int64:
		ns := v.Int64() * unitMul
		if ns < 0 {
			return 0, false
		}
		return uint64(ns), true
	case parquet.ByteArray, parquet.FixedLenByteArray:
		s := string(v.ByteArray())
		t, err := parseTLCDateString(s)
		if err != nil {
			return 0, false
		}
		ns := t.UnixNano()
		if ns < 0 {
			return 0, false
		}
		return uint64(ns), true
	}
	return 0, false
}

var tlcDateLayouts = []string{
	"2006-01-02 15:04:05",
	"2006-01-02T15:04:05",
	"2006-01-02 15:04:05.000",
	time.RFC3339,
}

func parseTLCDateString(s string) (time.Time, error) {
	s = strings.TrimSpace(s)
	for _, layout := range tlcDateLayouts {
		if t, err := time.Parse(layout, s); err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("unrecognized date format: %q", s)
}

// timestampUnitMultiplier returns the multiplier needed to convert raw int64
// timestamp values into nanoseconds. Defaults to microseconds (the TLC
// convention) when the logical type is missing or unrecognized.
func timestampUnitMultiplier(t parquet.Type) int64 {
	lt := t.LogicalType()
	if lt != nil && lt.Timestamp != nil {
		switch {
		case lt.Timestamp.Unit.Nanos != nil:
			return 1
		case lt.Timestamp.Unit.Micros != nil:
			return 1_000
		case lt.Timestamp.Unit.Millis != nil:
			return 1_000_000
		}
	}
	if ct := t.ConvertedType(); ct != nil {
		switch *ct {
		case deprecated.TimestampMicros:
			return 1_000
		case deprecated.TimestampMillis:
			return 1_000_000
		}
	}
	return 1_000
}
