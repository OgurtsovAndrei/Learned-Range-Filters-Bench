package snarf

/*
#cgo CXXFLAGS: -std=c++17
#cgo LDFLAGS: -L${SRCDIR}/build -lsnarf_wrapper -lc++ -lm
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// snarfKeyMask truncates inputs to 60 bits, the maximum width supported by
// SNARF's internal Golomb-Coded Set. Other filters in the harness handle full
// 64-bit keys, so the mask is applied here rather than at every call site.
const snarfKeyMask = (uint64(1) << 60) - 1

type SNARFFilter struct {
	ptr C.SNARFPtr
	n   int
}

func New(keys []uint64, bpk float64) *SNARFFilter {
	if len(keys) == 0 {
		return &SNARFFilter{}
	}
	masked := make([]uint64, len(keys))
	for i, k := range keys {
		masked[i] = k & snarfKeyMask
	}
	ptr := C.snarf_new((*C.uint64_t)(unsafe.Pointer(&masked[0])), C.size_t(len(masked)), C.double(bpk))
	f := &SNARFFilter{ptr: ptr, n: len(masked)}
	runtime.SetFinalizer(f, func(obj *SNARFFilter) {
		if obj.ptr != nil {
			C.snarf_free(obj.ptr)
			obj.ptr = nil
		}
	})
	return f
}

// IsEmpty returns true if the range [lo, hi] is definitely empty.
func (f *SNARFFilter) IsEmpty(lo, hi uint64) bool {
	if f.ptr == nil {
		return true
	}
	loM, hiM := lo&snarfKeyMask, hi&snarfKeyMask
	if hiM < loM {
		hiM = loM
	}
	return C.snarf_query(f.ptr, C.uint64_t(loM), C.uint64_t(hiM)) == 0
}

const queryBatchSize = 1024

func (f *SNARFFilter) QueryBatch(queries [][2]uint64) []bool {
	n := len(queries)
	result := make([]bool, n)
	if f.ptr == nil || n == 0 {
		for i := range result {
			result[i] = true
		}
		return result
	}
	masked := make([][2]uint64, n)
	for i, q := range queries {
		loM, hiM := q[0]&snarfKeyMask, q[1]&snarfKeyMask
		if hiM < loM {
			hiM = loM
		}
		masked[i] = [2]uint64{loM, hiM}
	}
	buf := make([]C.uint8_t, n)
	for off := 0; off < n; off += queryBatchSize {
		chunk := n - off
		if chunk > queryBatchSize {
			chunk = queryBatchSize
		}
		C.snarf_query_batch(f.ptr,
			(*C.uint64_t)(unsafe.Pointer(&masked[off][0])),
			C.size_t(chunk),
			(*C.uint8_t)(unsafe.Pointer(&buf[off])))
	}
	for i, v := range buf {
		result[i] = v == 0
	}
	return result
}

func (f *SNARFFilter) SizeInBits() uint64 {
	if f.ptr == nil {
		return 0
	}
	return uint64(C.snarf_size_bits(f.ptr))
}
