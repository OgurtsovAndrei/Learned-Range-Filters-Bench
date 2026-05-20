# ERE Cache Hit/Miss Measurement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-query hardware cache counter measurement (L1/LLC/instructions) comparing `ere` (two-vector) vs `ere_one_d` (one-vector) across all benchmark distributions.

**Architecture:** A pure-Go `perf_event_open` wrapper in `bench/internal/perf/perf_linux.go` uses `golang.org/x/sys/unix.PerfEventOpen` to open 5 hardware counters as a grouped fd, measuring all events atomically per query loop. `bench/ere_cache_test.go` runs `TestERECacheHitMiss` which warms up, measures 100k queries under the group, and emits a markdown table.

**Tech Stack:** Go 1.25, `golang.org/x/sys/unix` (already in go.mod as indirect dep), `//go:build linux` build tag.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `bench/internal/perf/perf_linux.go` | Create | EventSpec, Group, OpenGroup, Enable/Disable/Reset/Read/Close |
| `bench/internal/perf/perf_linux_test.go` | Create | Unit tests for perf wrapper |
| `bench/ere_cache_test.go` | Create | TestERECacheHitMiss — warmup + measure + markdown report |

---

## Key Constants (verified against golang.org/x/sys@v0.39.0)

```
unix.PERF_TYPE_HARDWARE           = 0
unix.PERF_TYPE_HW_CACHE           = 3
unix.PERF_COUNT_HW_INSTRUCTIONS   = 1
unix.PERF_COUNT_HW_CACHE_L1D      = 0
unix.PERF_COUNT_HW_CACHE_LL       = 2
unix.PERF_COUNT_HW_CACHE_OP_READ  = 0
unix.PERF_COUNT_HW_CACHE_RESULT_ACCESS = 0
unix.PERF_COUNT_HW_CACHE_RESULT_MISS   = 1
unix.PERF_FORMAT_TOTAL_TIME_ENABLED = 0x1
unix.PERF_FORMAT_TOTAL_TIME_RUNNING = 0x2
unix.PERF_FORMAT_GROUP              = 0x8
unix.PERF_EVENT_IOC_ENABLE  = 0x2400
unix.PERF_EVENT_IOC_DISABLE = 0x2401
unix.PERF_EVENT_IOC_RESET   = 0x2403
unix.PERF_FLAG_FD_CLOEXEC   = 0x8
unix.PerfBitDisabled         = CBitFieldMaskBit0 (== 1)

cache config encoding: cacheId | (cacheOp << 8) | (cacheResult << 16)
```

---

## Task 1: EventSpec, Group struct, OpenGroup, Close

**Files:**
- Create: `bench/internal/perf/perf_linux.go`
- Create: `bench/internal/perf/perf_linux_test.go`

- [ ] **Step 1: Write failing test**

Create `bench/internal/perf/perf_linux_test.go`:

```go
//go:build linux

package perf_test

import (
	"testing"

	"Thesis-bench-industry/bench/internal/perf"
)

func TestOpenGroupOpens(t *testing.T) {
	g, err := perf.OpenGroup([]perf.EventSpec{perf.Instructions})
	if err != nil {
		t.Skip("perf_event_open unavailable:", err)
	}
	g.Close()
}
```

- [ ] **Step 2: Run test — expect compile error (package doesn't exist yet)**

```bash
cd /home/andrei/Learned-Range-Filters-Bench
go test ./bench/internal/perf/ 2>&1 | head -5
```

Expected: `cannot find package` or build error.

- [ ] **Step 3: Create the implementation**

Create `bench/internal/perf/perf_linux.go`:

```go
//go:build linux

package perf

import (
	"encoding/binary"
	"fmt"
	"unsafe"

	"golang.org/x/sys/unix"
)

// EventSpec identifies a hardware performance counter.
type EventSpec struct {
	Type   uint32
	Config uint64
}

// Predefined event specs (cache config = cacheId | cacheOp<<8 | cacheResult<<16).
var (
	L1DLoads = EventSpec{
		unix.PERF_TYPE_HW_CACHE,
		uint64(unix.PERF_COUNT_HW_CACHE_L1D) |
			uint64(unix.PERF_COUNT_HW_CACHE_OP_READ)<<8 |
			uint64(unix.PERF_COUNT_HW_CACHE_RESULT_ACCESS)<<16,
	}
	L1DLoadMisses = EventSpec{
		unix.PERF_TYPE_HW_CACHE,
		uint64(unix.PERF_COUNT_HW_CACHE_L1D) |
			uint64(unix.PERF_COUNT_HW_CACHE_OP_READ)<<8 |
			uint64(unix.PERF_COUNT_HW_CACHE_RESULT_MISS)<<16,
	}
	LLCLoads = EventSpec{
		unix.PERF_TYPE_HW_CACHE,
		uint64(unix.PERF_COUNT_HW_CACHE_LL) |
			uint64(unix.PERF_COUNT_HW_CACHE_OP_READ)<<8 |
			uint64(unix.PERF_COUNT_HW_CACHE_RESULT_ACCESS)<<16,
	}
	LLCLoadMisses = EventSpec{
		unix.PERF_TYPE_HW_CACHE,
		uint64(unix.PERF_COUNT_HW_CACHE_LL) |
			uint64(unix.PERF_COUNT_HW_CACHE_OP_READ)<<8 |
			uint64(unix.PERF_COUNT_HW_CACHE_RESULT_MISS)<<16,
	}
	Instructions = EventSpec{
		unix.PERF_TYPE_HARDWARE,
		unix.PERF_COUNT_HW_INSTRUCTIONS,
	}
)

const (
	// perfReadFormat: read all group members + time stats in one read(2) call.
	perfReadFormat = uint64(unix.PERF_FORMAT_GROUP) |
		uint64(unix.PERF_FORMAT_TOTAL_TIME_ENABLED) |
		uint64(unix.PERF_FORMAT_TOTAL_TIME_RUNNING)

	// perfIOCFlagGroup: ioctl applies to all group members.
	perfIOCFlagGroup = uintptr(1)
)

// Group holds a set of hardware counters opened as a perf event group.
// All counters start/stop atomically; a single read on the leader fd
// returns all values.
type Group struct {
	fds []int
	n   int
}

// OpenGroup opens events as a perf event group on the current thread (pid=0),
// any CPU (cpu=-1). All counters start disabled; call Enable to begin counting.
//
// Returns an error wrapping unix.EPERM if perf_event_paranoid > 2.
// Caller must call Close when done.
func OpenGroup(events []EventSpec) (*Group, error) {
	if len(events) == 0 {
		return nil, fmt.Errorf("perf: need at least one event")
	}
	fds := make([]int, len(events))
	for i, e := range events {
		attr := unix.PerfEventAttr{
			Type:   e.Type,
			Config: e.Config,
			Bits:   unix.PerfBitDisabled,
		}
		attr.Size = uint32(unsafe.Sizeof(attr))
		groupFd := -1
		if i == 0 {
			// Leader: set read_format so Read() returns all members at once.
			attr.Read_format = perfReadFormat
		} else {
			groupFd = fds[0]
		}
		fd, err := unix.PerfEventOpen(&attr, 0, -1, groupFd, unix.PERF_FLAG_FD_CLOEXEC)
		if err != nil {
			for j := 0; j < i; j++ {
				unix.Close(fds[j])
			}
			if err == unix.EPERM || err == unix.EACCES {
				return nil, fmt.Errorf("perf: permission denied — set /proc/sys/kernel/perf_event_paranoid <= 2: %w", err)
			}
			return nil, fmt.Errorf("perf: open event %d (type=%d config=0x%x): %w", i, e.Type, e.Config, err)
		}
		fds[i] = fd
	}
	return &Group{fds: fds, n: len(events)}, nil
}

// Close releases all file descriptors.
func (g *Group) Close() {
	for _, fd := range g.fds {
		unix.Close(fd)
	}
}

// GroupResult holds raw counter values from one Read call.
type GroupResult struct {
	Values      []uint64 // Values[i] = count for events[i] passed to OpenGroup
	TimeEnabled uint64   // nanoseconds the group was enabled
	TimeRunning uint64   // nanoseconds the group was actually on the PMU
}

func ioctl(fd int, req uintptr, arg uintptr) error {
	_, _, errno := unix.Syscall(unix.SYS_IOCTL, uintptr(fd), req, arg)
	if errno != 0 {
		return errno
	}
	return nil
}

// Enable starts counting on all group members atomically.
func (g *Group) Enable() error {
	return ioctl(g.fds[0], unix.PERF_EVENT_IOC_ENABLE, perfIOCFlagGroup)
}

// Disable stops counting on all group members atomically.
func (g *Group) Disable() error {
	return ioctl(g.fds[0], unix.PERF_EVENT_IOC_DISABLE, perfIOCFlagGroup)
}

// Reset zeroes all counters in the group atomically.
func (g *Group) Reset() error {
	return ioctl(g.fds[0], unix.PERF_EVENT_IOC_RESET, perfIOCFlagGroup)
}

// Read returns all counter values in one atomic read from the leader fd.
// Buffer layout (PERF_FORMAT_GROUP | TIME_ENABLED | TIME_RUNNING):
//
//	u64 nr; u64 time_enabled; u64 time_running; u64 values[nr]
func (g *Group) Read() (GroupResult, error) {
	buf := make([]byte, (3+g.n)*8)
	n, err := unix.Read(g.fds[0], buf)
	if err != nil {
		return GroupResult{}, fmt.Errorf("perf: read: %w", err)
	}
	if n != len(buf) {
		return GroupResult{}, fmt.Errorf("perf: short read %d/%d bytes", n, len(buf))
	}
	nr := binary.LittleEndian.Uint64(buf[0:8])
	te := binary.LittleEndian.Uint64(buf[8:16])
	tr := binary.LittleEndian.Uint64(buf[16:24])
	values := make([]uint64, nr)
	for i := range values {
		values[i] = binary.LittleEndian.Uint64(buf[24+i*8:])
	}
	return GroupResult{Values: values, TimeEnabled: te, TimeRunning: tr}, nil
}
```

- [ ] **Step 4: Run test — expect PASS**

```bash
cd /home/andrei/Learned-Range-Filters-Bench
go test -v -run TestOpenGroupOpens ./bench/internal/perf/
```

Expected: `PASS` (or `SKIP` if perf_event_paranoid > 2 — that's acceptable).

- [ ] **Step 5: Commit**

```bash
git add bench/internal/perf/perf_linux.go bench/internal/perf/perf_linux_test.go
git commit -m "feat(bench): perf_event_open grouped fd wrapper (L1/LLC/instructions)"
```

---

## Task 2: Add GroupResult reading test (Enable/Disable/Reset/Read)

**Files:**
- Modify: `bench/internal/perf/perf_linux_test.go`

- [ ] **Step 1: Add failing test that reads instructions count**

Append to `bench/internal/perf/perf_linux_test.go`:

```go
func TestGroupReadInstructions(t *testing.T) {
	g, err := perf.OpenGroup([]perf.EventSpec{perf.Instructions})
	if err != nil {
		t.Skip("perf unavailable:", err)
	}
	defer g.Close()

	if err := g.Reset(); err != nil {
		t.Fatal("reset:", err)
	}
	if err := g.Enable(); err != nil {
		t.Fatal("enable:", err)
	}
	sum := 0
	for i := 0; i < 10_000; i++ {
		sum += i * i
	}
	if err := g.Disable(); err != nil {
		t.Fatal("disable:", err)
	}
	res, err := g.Read()
	if err != nil {
		t.Fatal("read:", err)
	}
	if len(res.Values) != 1 {
		t.Fatalf("expected 1 value, got %d", len(res.Values))
	}
	if res.Values[0] == 0 {
		t.Error("expected instructions > 0")
	}
	t.Logf("instructions for 10k iterations: %d (sum=%d)", res.Values[0], sum)
}

func TestGroupAllCacheEvents(t *testing.T) {
	events := []perf.EventSpec{
		perf.L1DLoads,
		perf.L1DLoadMisses,
		perf.LLCLoads,
		perf.LLCLoadMisses,
		perf.Instructions,
	}
	g, err := perf.OpenGroup(events)
	if err != nil {
		t.Skip("perf unavailable:", err)
	}
	defer g.Close()

	if err := g.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := g.Enable(); err != nil {
		t.Fatal(err)
	}
	// Touch 1 MB to generate L1/LLC loads.
	data := make([]byte, 1<<20)
	chk := byte(0)
	for _, b := range data {
		chk += b
	}
	if err := g.Disable(); err != nil {
		t.Fatal(err)
	}
	res, err := g.Read()
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Values) != 5 {
		t.Fatalf("expected 5 values, got %d", len(res.Values))
	}
	t.Logf("L1-loads=%d L1-misses=%d LLC-loads=%d LLC-misses=%d instrs=%d (chk=%d)",
		res.Values[0], res.Values[1], res.Values[2], res.Values[3], res.Values[4], chk)
	if res.Values[0] == 0 {
		t.Error("L1D loads should be > 0 after reading 1MB")
	}
	if res.Values[4] == 0 {
		t.Error("instructions should be > 0")
	}
}
```

- [ ] **Step 2: Run tests — expect PASS (or SKIP)**

```bash
cd /home/andrei/Learned-Range-Filters-Bench
go test -v -run "TestGroupRead|TestGroupAllCache" ./bench/internal/perf/
```

Expected output contains lines like:
```
--- PASS: TestGroupReadInstructions
    perf_linux_test.go:NN: instructions for 10k iterations: NNNN
--- PASS: TestGroupAllCacheEvents
    perf_linux_test.go:NN: L1-loads=NNNN L1-misses=NNN LLC-loads=NNN ...
```

- [ ] **Step 3: Commit**

```bash
git add bench/internal/perf/perf_linux_test.go
git commit -m "test(bench): perf wrapper integration tests — read/enable/disable/reset"
```

---

## Task 3: TestERECacheHitMiss — uniform distribution

**Files:**
- Create: `bench/ere_cache_test.go`

- [ ] **Step 1: Write TestERECacheHitMiss for uniform only**

Create `bench/ere_cache_test.go`:

```go
//go:build linux

package bench_test

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"Thesis-bench-industry/bench/internal/perf"
)

const (
	ereCacheQueryCount  = 100_000
	ereCacheWarmupCount = 10_000
)

var allPerfEvents = []perf.EventSpec{
	perf.L1DLoads,
	perf.L1DLoadMisses,
	perf.LLCLoads,
	perf.LLCLoadMisses,
	perf.Instructions,
}

type ereCacheRow struct {
	dataset    string
	filter     string
	l1Loads    float64
	l1Misses   float64
	l1MissRate float64 // percent
	llcLoads   float64
	llcMisses  float64
	llcMissRate float64 // percent
	instrs     float64
}

func measureERECacheEvents(t *testing.T, filter ereExactFilter, queries []ereQuery) ereCacheRow {
	t.Helper()

	// Warmup: bring filter data structures into cache, stabilise branch predictor.
	for i := 0; i < ereCacheWarmupCount; i++ {
		q := queries[i%len(queries)]
		ereQuerySink = filter.IsEmpty(q.a, q.b)
	}

	g, err := perf.OpenGroup(allPerfEvents)
	if err != nil {
		t.Fatal("open perf group:", err)
	}
	defer g.Close()

	if err := g.Reset(); err != nil {
		t.Fatal("reset:", err)
	}
	if err := g.Enable(); err != nil {
		t.Fatal("enable:", err)
	}
	for i := 0; i < ereCacheQueryCount; i++ {
		q := queries[i%len(queries)]
		ereQuerySink = filter.IsEmpty(q.a, q.b)
	}
	if err := g.Disable(); err != nil {
		t.Fatal("disable:", err)
	}

	res, err := g.Read()
	if err != nil {
		t.Fatal("read:", err)
	}

	n := float64(ereCacheQueryCount)
	l1Loads  := float64(res.Values[0]) / n
	l1Misses := float64(res.Values[1]) / n
	llcLoads  := float64(res.Values[2]) / n
	llcMisses := float64(res.Values[3]) / n
	instrs   := float64(res.Values[4]) / n

	l1Rate  := 0.0
	if l1Loads > 0 {
		l1Rate = l1Misses / l1Loads * 100
	}
	llcRate := 0.0
	if llcLoads > 0 {
		llcRate = llcMisses / llcLoads * 100
	}

	return ereCacheRow{
		l1Loads: l1Loads, l1Misses: l1Misses, l1MissRate: l1Rate,
		llcLoads: llcLoads, llcMisses: llcMisses, llcMissRate: llcRate,
		instrs: instrs,
	}
}

func TestERECacheHitMiss(t *testing.T) {
	// Probe perf availability before loading large datasets.
	probe, err := perf.OpenGroup([]perf.EventSpec{perf.Instructions})
	if err != nil {
		t.Skip("perf_event_open unavailable:", err)
	}
	probe.Close()

	datasets := mustLoadEREDatasets(t)

	var rows []ereCacheRow

	for _, ds := range datasets {
		ereFilter, err := buildEREFilter(ds.keys)
		if err != nil {
			t.Fatalf("%s: build ere: %v", ds.name, err)
		}
		oneDFilter, err := buildEREOneDFilter(ds.keys)
		if err != nil {
			t.Fatalf("%s: build ere_one_d: %v", ds.name, err)
		}

		r1 := measureERECacheEvents(t, ereFilter, ds.queries)
		r1.dataset = ds.name
		r1.filter = "ere"

		r2 := measureERECacheEvents(t, oneDFilter, ds.queries)
		r2.dataset = ds.name
		r2.filter = "ere_one_d"

		rows = append(rows, r1, r2)
	}

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "# ERE vs ERE One-D — Cache Events per Query\n\n")
	fmt.Fprintf(&buf, "n=%d, queries=%d (warmup=%d)\n\n",
		ereCompareN, ereCacheQueryCount, ereCacheWarmupCount)
	fmt.Fprintf(&buf, "| Dataset | Filter | L1-loads/q | L1-misses/q | L1-miss%% | LLC-loads/q | LLC-misses/q | LLC-miss%% | Instrs/q |\n")
	fmt.Fprintf(&buf, "|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, r := range rows {
		fmt.Fprintf(&buf, "| %s | %s | %.1f | %.2f | %.1f%% | %.2f | %.2f | %.1f%% | %.1f |\n",
			r.dataset, r.filter,
			r.l1Loads, r.l1Misses, r.l1MissRate,
			r.llcLoads, r.llcMisses, r.llcMissRate,
			r.instrs)
	}
	fmt.Print("\n" + buf.String())

	reportPath := filepath.Join("..", "bench_results", "ere_cache_report.md")
	if err := os.MkdirAll(filepath.Dir(reportPath), 0755); err == nil {
		if err := os.WriteFile(reportPath, buf.Bytes(), 0644); err != nil {
			t.Logf("warning: could not write report to %s: %v", reportPath, err)
		}
	}
}
```

- [ ] **Step 2: Run test — expect compile success and PASS/SKIP**

```bash
cd /home/andrei/Learned-Range-Filters-Bench
go test -v -run TestERECacheHitMiss -timeout 10m ./bench/
```

Expected (perf available): test prints markdown table with numbers, then `PASS`.
Expected (perf unavailable): `SKIP perf_event_open unavailable: perf: permission denied...`

If SKIP due to paranoid:
```bash
cat /proc/sys/kernel/perf_event_paranoid
# If > 2:
echo 2 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

- [ ] **Step 3: Verify output looks reasonable**

Check printed table:
- `L1-loads/q` should be in range 10–1000 (ERE does multiple Rank/Select calls)
- `LLC-misses/q` should be < 5 for uniform (filter fits in L2/L3)
- `Instrs/q` should be in range 100–2000
- `ere_one_d` LLC-miss% should be ≤ `ere` LLC-miss% (main thesis claim to verify)

- [ ] **Step 4: Commit**

```bash
git add bench/ere_cache_test.go
git commit -m "feat(bench): TestERECacheHitMiss — per-query L1/LLC/instr counters via perf_event_open"
```

---

## Self-Review Checklist

- [x] Spec coverage: OpenGroup ✓, 5 events ✓, warmup ✓, 100k measure ✓, all distributions (via mustLoadEREDatasets) ✓, markdown table ✓, bench_results file ✓, SKIP on unavailable perf ✓
- [x] No placeholders: all code blocks are complete
- [x] Type consistency: `ereCacheRow` used in Task 3 is defined in Task 3 ✓; `ereExactFilter`, `ereQuerySink`, `mustLoadEREDatasets`, `buildEREFilter`, `buildEREOneDFilter`, `ereCompareN` all come from `ere_compare_test.go` (same package `bench_test`) ✓
- [x] `allPerfEvents` index mapping: Values[0]=L1DLoads, [1]=L1DLoadMisses, [2]=LLCLoads, [3]=LLCLoadMisses, [4]=Instructions — matches slice order ✓
- [x] `PERF_EVENT_IOC_RESET = 0x2403` (verified, not 0x2404) ✓
- [x] `PERF_FORMAT_TOTAL_TIME_ENABLED = 0x1` (not 0x2) ✓
- [x] Read buffer size: `(3 + n) * 8` bytes for `nr + time_enabled + time_running + values[n]` ✓
