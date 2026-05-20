package rosetta

/*
#cgo CXXFLAGS: -std=c++17
#cgo LDFLAGS: -L${SRCDIR}/build -lrosetta_wrapper -lstdc++ -lm
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// RosettaFilter wraps a DstFilter<BloomFilter<>, false> built via the Rosetta
// C ABI. The construction parameters (bpk + sample query distribution) are
// captured at New() and cannot be changed after the fact.
type RosettaFilter struct {
	ptr C.RosettaPtr
	n   int
}

// New builds a Rosetta range-emptiness filter targeting `bpk` bits per key.
// `sampleLeft` and `sampleRight` are paired endpoints of representative
// queries (closed range semantics — the wrapper translates internally).
// Pass empty slices if no query distribution is available; this disables
// Rosetta's per-level BPK shaping. Both slices must have equal length.
func New(keys []uint64, bpk float64, sampleLeft, sampleRight []uint64) *RosettaFilter {
	if len(keys) == 0 {
		return &RosettaFilter{}
	}
	if len(sampleLeft) != len(sampleRight) {
		panic("rosetta.New: sampleLeft/sampleRight length mismatch")
	}

	var leftPtr, rightPtr *C.uint64_t
	if len(sampleLeft) > 0 {
		leftPtr = (*C.uint64_t)(unsafe.Pointer(&sampleLeft[0]))
		rightPtr = (*C.uint64_t)(unsafe.Pointer(&sampleRight[0]))
	}

	ptr := C.rosetta_new(
		(*C.uint64_t)(unsafe.Pointer(&keys[0])), C.size_t(len(keys)),
		C.double(bpk),
		leftPtr, rightPtr, C.size_t(len(sampleLeft)),
	)
	f := &RosettaFilter{ptr: ptr, n: len(keys)}
	runtime.SetFinalizer(f, func(obj *RosettaFilter) {
		if obj.ptr != nil {
			C.rosetta_free(obj.ptr)
			obj.ptr = nil
		}
	})
	return f
}

// IsEmpty returns true if the closed range [lo, hi] is definitely empty.
func (f *RosettaFilter) IsEmpty(lo, hi uint64) bool {
	if f.ptr == nil {
		return true
	}
	return C.rosetta_query(f.ptr, C.uint64_t(lo), C.uint64_t(hi)) == 0
}

const queryBatchSize = 1024

func (f *RosettaFilter) QueryBatch(queries [][2]uint64) []bool {
	n := len(queries)
	result := make([]bool, n)
	if f.ptr == nil || n == 0 {
		for i := range result {
			result[i] = true
		}
		return result
	}
	buf := make([]C.uint8_t, n)
	for off := 0; off < n; off += queryBatchSize {
		chunk := n - off
		if chunk > queryBatchSize {
			chunk = queryBatchSize
		}
		C.rosetta_query_batch(f.ptr,
			(*C.uint64_t)(unsafe.Pointer(&queries[off][0])),
			C.size_t(chunk),
			(*C.uint8_t)(unsafe.Pointer(&buf[off])))
	}
	for i, v := range buf {
		result[i] = v == 0
	}
	return result
}

func (f *RosettaFilter) SizeInBits() uint64 {
	if f.ptr == nil {
		return 0
	}
	return uint64(C.rosetta_size_bits(f.ptr))
}
