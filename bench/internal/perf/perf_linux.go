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

// GroupResult holds raw counter values from one Read call.
type GroupResult struct {
	Values      []uint64 // Values[i] = count for events[i] passed to OpenGroup
	TimeEnabled uint64   // nanoseconds the group was enabled
	TimeRunning uint64   // nanoseconds the group was actually on the PMU
}

// OpenGroup opens events as a perf event group on the current process (pid=0),
// any CPU (cpu=-1). All counters start disabled; call Enable to begin counting.
//
// Returns a wrapped unix.EPERM error if perf_event_paranoid > 2.
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
