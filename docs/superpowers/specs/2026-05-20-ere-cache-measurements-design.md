# ERE vs ERE-OneD Cache Hit/Miss Measurement

**Date:** 2026-05-20  
**Goal:** Quantify how the two-vector (D1, D2) → one-vector (D) optimization in ERE changes L1/LLC cache behaviour, per query, across all benchmark distributions.

---

## Background

`ere.tex` §4 ("One-Vector Optimization") claims that concentrating Rank/Select calls onto a single contiguous bit-vector D improves cache locality: cache lines loaded by the first Select are likely to serve subsequent Selects in the same query. This claim is currently qualitative. We now have a Linux machine (kernel 6.17) and can validate it with hardware performance counters.

---

## Scope

- **Filters under test:** `Thesis/emptiness/exact/ere` (two-vector) vs `Thesis/emptiness/exact/ere_one_d` (one-vector).
- **Distributions:** uniform, clustered, sosd_fb, sosd_wiki, sosd_osm, sosd_books — same set as `ere_compare_test.go`.
- **N:** 2^20 keys (same as `ereCompareN`).
- **Events measured per query:** L1D-loads, L1D-load-misses, LLC-loads, LLC-load-misses, HW-instructions.

---

## Architecture

### 1. `bench/internal/perf/perf_linux.go`

Pure Go (no CGo). Uses `syscall.Syscall6(SYS_PERF_EVENT_OPEN, ...)`.

**Structs:**
```
PerfEventAttr  — mirrors kernel struct perf_event_attr (64-byte, packed)
GroupResult    — []uint64 of length = number of group members, plus time_enabled / time_running
Group          — holds leader fd + member fds
```

**API:**
```
type EventSpec struct { Type uint32; Config uint64 }

func OpenGroup(pid int, events []EventSpec) (*Group, error)
  // pid=0 means current thread; cpu=-1 means any CPU
  // First event becomes leader (groupFd=-1), rest are members (groupFd=leader)
  // All opened with PERF_FLAG_FD_CLOEXEC

func (*Group) Enable() error   // ioctl(PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP)
func (*Group) Disable() error  // ioctl(PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP)
func (*Group) Reset() error    // ioctl(PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP)
func (*Group) Read() (GroupResult, error)  // read leader fd with PERF_FORMAT_GROUP|PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING
func (*Group) Close()
```

**Predefined EventSpecs (package-level vars):**
```go
var (
    L1DLoads      = EventSpec{PERF_TYPE_HW_CACHE, l1dReadAccess}
    L1DLoadMisses = EventSpec{PERF_TYPE_HW_CACHE, l1dReadMiss}
    LLCLoads      = EventSpec{PERF_TYPE_HW_CACHE, llcReadAccess}
    LLCLoadMisses = EventSpec{PERF_TYPE_HW_CACHE, llcReadMiss}
    Instructions  = EventSpec{PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS}
)
```

Cache config encoding:
```
config = cacheId | (cacheOp << 8) | (cacheResult << 16)
```

**Error handling:** if a hardware event is unsupported (returns ENOENT or EOPNOTSUPP), `OpenGroup` returns a descriptive error. The test skips on `t.Skip` if perf is unavailable (e.g. `perf_event_paranoid > 2`).

---

### 2. `bench/ere_cache_test.go`

**Test function:** `TestERECacheHitMiss(t *testing.T)`

**Per dataset, per filter:**
1. Build filter (ere / ere_one_d) — same helpers as `ere_compare_test.go`
2. Warmup: run 10 000 queries without counters (bring filter into cache, stabilise branch predictor)
3. Open perf group (5 events, pid=0 current thread)
4. Reset + Enable
5. Run 100 000 queries (same query slice, round-robin)
6. Disable + Read → GroupResult
7. Compute per-query metrics:
   - `l1_loads_per_q  = result[0] / 100000`
   - `l1_misses_per_q = result[1] / 100000`
   - `llc_loads_per_q = result[2] / 100000`
   - `llc_misses_per_q= result[3] / 100000`
   - `instrs_per_q    = result[4] / 100000`
   - `l1_miss_rate_%  = l1_misses / l1_loads * 100`
   - `llc_miss_rate_% = llc_misses / llc_loads * 100`
8. Close group

**Output:** markdown table printed to stdout and written to `bench_results/ere_cache_report.md`:

| Dataset | Filter | L1-loads/q | L1-misses/q | L1-miss% | LLC-loads/q | LLC-misses/q | LLC-miss% | Instrs/q |
|---|---|---:|---:|---:|---:|---:|---:|---:|

**No new test helpers beyond what ere_compare_test.go already exports** — reuse `mustLoadEREDatasets`, `buildEREFilter`, `buildEREOneDFilter`, `ereQuerySink`.

---

## Error / Skip Policy

- `perf_event_paranoid > 2`: `t.Skip("perf_event_open requires paranoid <= 2")`
- Any event ENOENT: `t.Skip("hardware cache counters not available on this CPU")`
- Other syscall error: `t.Fatal`

---

## Files Changed

| File | Action |
|---|---|
| `bench/internal/perf/perf_linux.go` | New |
| `bench/ere_cache_test.go` | New |

No existing files modified.

---

## Out of Scope

- Windows / macOS support (Linux-only via `//go:build linux`)
- CGo — pure Go syscall only
- Write events, branch misses, TLB misses
- Grafite / SNARF / SuRF cache comparison (separate task)
