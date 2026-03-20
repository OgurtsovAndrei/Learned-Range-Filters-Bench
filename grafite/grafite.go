package grafite

/*
#cgo CXXFLAGS: -std=c++17
#cgo LDFLAGS: -L${SRCDIR}/build -lgrafite_wrapper -lc++ -lm
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

type GrafiteFilter struct {
	ptr C.GrafitePtr
	n   int
}

func New(keys []uint64, bpk float64) *GrafiteFilter {
	if len(keys) == 0 {
		return &GrafiteFilter{}
	}
	ptr := C.grafite_new((*C.uint64_t)(unsafe.Pointer(&keys[0])), C.size_t(len(keys)), C.double(bpk))
	f := &GrafiteFilter{ptr: ptr, n: len(keys)}
	runtime.SetFinalizer(f, func(obj *GrafiteFilter) {
		if obj.ptr != nil {
			C.grafite_free(obj.ptr)
			obj.ptr = nil
		}
	})
	return f
}

// IsEmpty returns true if the range [lo, hi] is definitely empty.
func (f *GrafiteFilter) IsEmpty(lo, hi uint64) bool {
	if f.ptr == nil {
		return true
	}
	return C.grafite_query(f.ptr, C.uint64_t(lo), C.uint64_t(hi)) == 0
}

const queryBatchSize = 1024

func (f *GrafiteFilter) QueryBatch(queries [][2]uint64) []bool {
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
		C.grafite_query_batch(f.ptr,
			(*C.uint64_t)(unsafe.Pointer(&queries[off][0])),
			C.size_t(chunk),
			(*C.uint8_t)(unsafe.Pointer(&buf[off])))
	}
	for i, v := range buf {
		result[i] = v == 0
	}
	return result
}

func (f *GrafiteFilter) SizeInBits() uint64 {
	if f.ptr == nil {
		return 0
	}
	return uint64(C.grafite_size_bits(f.ptr))
}
