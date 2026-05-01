// b6_cache_test.go — caching layer for B6 sweep results.
//
// Each (n, queryCount, eps) run owns a directory
// `bench_results/data/b6_latency_N{N}/` containing one JSON file per
// filter (`Grafite.json`, `SODA.json`, ...) plus a global header
// `_meta.json`. Per-filter files allow surgical invalidation: to drop
// stale data for a single filter family, `rm <Filter>.json` is enough.
//
// b6Store is the in-memory writer. It loads per-filter docs lazily on
// first access, performs a one-shot migration from the pre-refactor
// monolithic `b6_latency_N{N}.json` (renamed to `*.legacy` after
// migration), and atomically persists each touched filter on flush
// (write-to-tmp + rename).
//
// The runner in b6_latency_test.go interacts with the cache via
// update / cachedRow / flush only — all on-disk shape lives here.

package bench_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// b6SchemaVersion is the per-filter doc schema version. Bump when row
// shape changes incompatibly so older caches are dropped or migrated.
const b6SchemaVersion = 2

// ----------------------------------------------------------------------------
// Row + parameter hashing
// ----------------------------------------------------------------------------

type b6Row struct {
	Distribution  string  `json:"distribution"`
	Filter        string  `json:"filter"`
	RangeLen      uint64  `json:"rangeLen"`
	BuildNs       int64   `json:"buildNs"`
	BuildMKeysSec float64 `json:"buildMKeysSec"`
	QueryNsPerOp  float64 `json:"queryNsPerOp"` // wall-clock ns / total queries (with parallelism)
	BPKUsed       float64 `json:"bpkUsed"`
	SizeBits      uint64  `json:"sizeBits"`
	FPR           float64 `json:"fpr"`
	QueriesEmpty  int     `json:"queriesEmpty"`
	SweepName     string  `json:"sweepName"`
	SweepParam    float64 `json:"sweepParam"`
	Parallelism   int     `json:"parallelism"`
	ParamsHash    string  `json:"paramsHash"`
	Note          string  `json:"note,omitempty"`
}

// b6Params captures the hyperparameters that, if changed, invalidate a
// cached row. Stored hashed into b6Row.ParamsHash; the runner computes the
// hash for the current run and skips rows whose stored hash matches.
type b6Params struct {
	NKeys         int     `json:"nKeys"`
	Eps           float64 `json:"eps"`
	RangeLen      uint64  `json:"rangeLen"`
	QueryCount    int     `json:"queryCount"`
	QueryStrategy string  `json:"queryStrategy"`
	QuerySeed     int64   `json:"querySeed"`
	Distribution  string  `json:"distribution"`
	Filter        string  `json:"filter"`
	SweepName     string  `json:"sweepName"`
	SweepParam    float64 `json:"sweepParam"`
	Parallelism   int     `json:"parallelism"`
}

func (p b6Params) hash() string {
	buf, _ := json.Marshal(p)
	sum := sha256.Sum256(buf)
	return hex.EncodeToString(sum[:8]) // 16 hex chars is plenty for collisions
}

// ----------------------------------------------------------------------------
// On-disk shapes
// ----------------------------------------------------------------------------

// b6Doc preserves the legacy single-file layout. It is still used by the
// plotter as an aggregation type and by the migration code path that reads
// a pre-refactor `b6_latency_N{N}.json` and splits its rows per filter.
type b6Doc struct {
	Type          string  `json:"type"`
	NKeys         int     `json:"nKeys"`
	QueryCount    int     `json:"queryCount"`
	QueryStrategy string  `json:"queryStrategy"`
	Eps           float64 `json:"eps"`
	Timestamp     string  `json:"timestamp"`
	GitCommit     string  `json:"gitCommit"`
	Rows          []b6Row `json:"rows"`
}

// b6FilterDoc is the on-disk shape of a per-filter cache file. One file
// per filter under bench_results/data/b6_latency_N{N}/<filter>.json.
type b6FilterDoc struct {
	Type          string  `json:"type"` // "b6_latency_filter"
	Filter        string  `json:"filter"`
	NKeys         int     `json:"nKeys"`
	QueryCount    int     `json:"queryCount"`
	QueryStrategy string  `json:"queryStrategy"`
	Eps           float64 `json:"eps"`
	Timestamp     string  `json:"timestamp"`
	GitCommit     string  `json:"gitCommit"`
	SchemaVersion int     `json:"schemaVersion"`
	Rows          []b6Row `json:"rows"`
}

// b6MetaDoc is the small global header written alongside the per-filter
// files as `_meta.json`. It captures run-level invariants (n, queryCount,
// eps, queryStrategy) shared across all filters.
type b6MetaDoc struct {
	Type          string  `json:"type"` // "b6_latency"
	NKeys         int     `json:"nKeys"`
	QueryCount    int     `json:"queryCount"`
	QueryStrategy string  `json:"queryStrategy"`
	Eps           float64 `json:"eps"`
	SchemaVersion int     `json:"schemaVersion"`
	CreatedAt     string  `json:"createdAt"`
}

// ----------------------------------------------------------------------------
// b6Store
// ----------------------------------------------------------------------------

// b6Store is the per-(distribution,filter) incremental writer. Each filter
// has its own JSON file under bench_results/data/b6_latency_N{N}/. Files
// are loaded lazily on first access (read-on-touch via loadFilterLocked)
// so a store covering many filters does no I/O until needed. Legacy rows
// lacking SweepName are dropped on load — they predate the K-sweep
// schema and would clutter plots.
type b6Store struct {
	mu            sync.Mutex
	nKeys         int
	queryCount    int
	eps           float64
	queryStrategy string
	// mixSuffix is appended to the on-disk path for non-default workload
	// variants (e.g. "gap_heavy"). Empty for the historical 50/30/20 mix
	// so its layout — bench_results/data/b6_latency_N{N}/ — is preserved.
	mixSuffix string
	// filters is keyed by filter name. A nil entry (key not present) means
	// the per-filter doc has not been loaded yet; a non-nil entry has been
	// loaded (possibly empty — file did not exist on disk). This matters
	// because we want to flush "loaded but never updated" filters as no-op,
	// and not accidentally re-migrate.
	filters map[string]*b6FilterDoc
	// dirty marks per-filter docs that have been mutated (update or
	// migrated from legacy) and need persistence on flush.
	dirty map[string]bool
	// migrated records whether a one-shot legacy migration has already
	// been attempted, so subsequent loadFilterLocked calls don't redo it.
	migrated bool
}

// newB6Store builds a store keyed at (nKeys, queryStrategy). mixSuffix is
// appended to the on-disk path for non-default workload variants — empty
// keeps the historical layout, "gap_heavy" routes to a sibling dir.
func newB6Store(nKeys, queryCount int, eps float64, queryStrategy, mixSuffix string) *b6Store {
	return &b6Store{
		nKeys:         nKeys,
		queryCount:    queryCount,
		eps:           eps,
		queryStrategy: queryStrategy,
		mixSuffix:     mixSuffix,
		filters:       make(map[string]*b6FilterDoc),
		dirty:         make(map[string]bool),
	}
}

// path returns the per-N directory holding all per-filter cache files
// plus _meta.json. It does not guarantee the directory exists; flush()
// creates it on demand.
func (s *b6Store) path() string {
	if s.mixSuffix == "" {
		return fmt.Sprintf("../bench_results/data/b6_latency_N%d", s.nKeys)
	}
	return fmt.Sprintf("../bench_results/data/b6_latency_N%d_%s", s.nKeys, s.mixSuffix)
}

// legacyPath is the pre-refactor monolithic file location. Used only by
// the one-shot migration in migrateLegacyLocked. Non-default workload
// variants never had a legacy file, so we return an empty path which the
// migrator treats as a no-op.
func (s *b6Store) legacyPath() string {
	if s.mixSuffix != "" {
		return ""
	}
	return fmt.Sprintf("../bench_results/data/b6_latency_N%d.json", s.nKeys)
}

// filterFilePath is the per-filter cache file under path().
func (s *b6Store) filterFilePath(filter string) string {
	return filepath.Join(s.path(), filter+".json")
}

// metaFilePath is the global header file under path().
func (s *b6Store) metaFilePath() string {
	return filepath.Join(s.path(), "_meta.json")
}

// loadFilterLocked lazily populates s.filters[name]. On first access for
// any filter it also performs the one-shot legacy single-file migration:
// reads `b6_latency_N{N}.json` (if present), splits its rows per Filter
// field, populates s.filters with the migrated docs (marked dirty so
// flush persists them), and renames the legacy file to `*.legacy` so
// subsequent runs don't re-migrate.
//
// Caller must hold s.mu.
func (s *b6Store) loadFilterLocked(name string) *b6FilterDoc {
	// One-shot legacy migration on first access. Independent of which
	// filter is requested: a single legacy file holds rows for many
	// filters and we want them all visible immediately.
	if !s.migrated {
		s.migrated = true
		s.migrateLegacyLocked()
	}
	if doc, ok := s.filters[name]; ok {
		return doc
	}
	doc := &b6FilterDoc{
		Type:          "b6_latency_filter",
		Filter:        name,
		NKeys:         s.nKeys,
		QueryCount:    s.queryCount,
		QueryStrategy: s.queryStrategy,
		Eps:           s.eps,
		SchemaVersion: b6SchemaVersion,
	}
	if buf, err := os.ReadFile(s.filterFilePath(name)); err == nil {
		var prior b6FilterDoc
		if err := json.Unmarshal(buf, &prior); err == nil {
			doc.Type = prior.Type
			if doc.Type == "" {
				doc.Type = "b6_latency_filter"
			}
			doc.Timestamp = prior.Timestamp
			doc.GitCommit = prior.GitCommit
			if prior.SchemaVersion != 0 {
				doc.SchemaVersion = prior.SchemaVersion
			}
			for _, r := range prior.Rows {
				if r.SweepName == "" {
					continue
				}
				doc.Rows = append(doc.Rows, r)
			}
		}
	}
	s.filters[name] = doc
	return doc
}

// migrateLegacyLocked reads the legacy single-file format if it exists,
// splits its rows per filter into per-filter docs, and renames the legacy
// file to <path>.legacy so the migration is one-shot. Filters populated
// by the migration are marked dirty so flush persists them. Caller must
// hold s.mu.
func (s *b6Store) migrateLegacyLocked() {
	legacy := s.legacyPath()
	buf, err := os.ReadFile(legacy)
	if err != nil {
		return // no legacy file, nothing to migrate
	}
	var prior b6Doc
	if err := json.Unmarshal(buf, &prior); err != nil {
		return
	}
	grouped := make(map[string][]b6Row)
	for _, r := range prior.Rows {
		if r.SweepName == "" {
			continue // legacy pre-K-sweep row
		}
		if r.Filter == "" {
			continue
		}
		grouped[r.Filter] = append(grouped[r.Filter], r)
	}
	for filter, rows := range grouped {
		// If a per-filter file already exists on disk, prefer the on-disk
		// rows (newer format) and skip migration for this filter. We
		// detect this by stat'ing the file rather than re-reading it
		// — this codepath runs before normal loadFilterLocked.
		if _, statErr := os.Stat(s.filterFilePath(filter)); statErr == nil {
			continue
		}
		doc := &b6FilterDoc{
			Type:          "b6_latency_filter",
			Filter:        filter,
			NKeys:         prior.NKeys,
			QueryCount:    prior.QueryCount,
			QueryStrategy: prior.QueryStrategy,
			Eps:           prior.Eps,
			Timestamp:     prior.Timestamp,
			GitCommit:     prior.GitCommit,
			SchemaVersion: b6SchemaVersion,
			Rows:          rows,
		}
		s.filters[filter] = doc
		s.dirty[filter] = true
	}
	// Rename legacy file so we don't re-migrate next time. `.legacy`
	// suffix lets the user recover if anything goes wrong.
	_ = os.Rename(legacy, legacy+".legacy")
}

// update merges new measurement rows for (dist, filter) into the
// per-filter doc. Rows whose paramsHash matches one of the incoming rows
// are replaced; rows with different paramsHash (e.g. different
// parallelism, different sweep grid) are preserved so a run at
// B6_PARALLEL=4 does not clobber the P=1 data.
func (s *b6Store) update(dist, filter string, rows []b6Row) {
	s.mu.Lock()
	defer s.mu.Unlock()
	doc := s.loadFilterLocked(filter)
	incoming := make(map[string]struct{}, len(rows))
	for _, r := range rows {
		incoming[r.ParamsHash] = struct{}{}
	}
	kept := doc.Rows[:0]
	for _, r := range doc.Rows {
		if r.Distribution == dist && r.Filter == filter {
			if _, replaced := incoming[r.ParamsHash]; replaced {
				continue
			}
		}
		kept = append(kept, r)
	}
	doc.Rows = append(kept, rows...)
	s.dirty[filter] = true
}

// cachedRow returns a prior row for (dist, filter, L, sweepName, sweepParam)
// whose paramsHash matches the requested one, or nil. Use to short-circuit
// measurement when FORCE is unset.
func (s *b6Store) cachedRow(dist, filter string, L uint64, sweepName string, sweepParam float64, paramsHash string) *b6Row {
	s.mu.Lock()
	defer s.mu.Unlock()
	doc := s.loadFilterLocked(filter)
	for i := range doc.Rows {
		r := &doc.Rows[i]
		if r.Distribution == dist && r.Filter == filter &&
			r.RangeLen == L && r.SweepName == sweepName &&
			r.SweepParam == sweepParam && r.ParamsHash == paramsHash {
			return r
		}
	}
	return nil
}

// flush persists every dirty per-filter doc plus _meta.json under
// path(). Each file is written atomically (write-to-tmp + rename) so an
// aborted run cannot leave a half-written cache. The directory is
// created on demand.
func (s *b6Store) flush() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := os.MkdirAll(s.path(), 0755); err != nil {
		return err
	}
	now := time.Now().UTC().Format(time.RFC3339)
	commit := gitCommitShort()
	for name, doc := range s.filters {
		if !s.dirty[name] {
			continue
		}
		doc.Timestamp = now
		doc.GitCommit = commit
		doc.SchemaVersion = b6SchemaVersion
		doc.Type = "b6_latency_filter"
		buf, err := json.MarshalIndent(doc, "", "  ")
		if err != nil {
			return err
		}
		if err := writeFileAtomic(s.filterFilePath(name), buf, 0644); err != nil {
			return err
		}
		s.dirty[name] = false
	}
	meta := b6MetaDoc{
		Type:          "b6_latency",
		NKeys:         s.nKeys,
		QueryCount:    s.queryCount,
		QueryStrategy: s.queryStrategy,
		Eps:           s.eps,
		SchemaVersion: b6SchemaVersion,
		CreatedAt:     now,
	}
	mbuf, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	return writeFileAtomic(s.metaFilePath(), mbuf, 0644)
}

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------

// writeFileAtomic writes data to path via a sibling tmp file and rename,
// so an aborted run can't leave a half-written cache.
func writeFileAtomic(path string, data []byte, perm os.FileMode) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, perm); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}
