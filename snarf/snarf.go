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

type SNARFFilter struct {
	ptr C.SNARFPtr
	n   int
}

func New(keys []uint64, bpk float64) *SNARFFilter {
	if len(keys) == 0 {
		return &SNARFFilter{}
	}
	ptr := C.snarf_new((*C.uint64_t)(unsafe.Pointer(&keys[0])), C.size_t(len(keys)), C.double(bpk))
	f := &SNARFFilter{ptr: ptr, n: len(keys)}
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
	return C.snarf_query(f.ptr, C.uint64_t(lo), C.uint64_t(hi)) == 0
}

func (f *SNARFFilter) SizeInBits() uint64 {
	if f.ptr == nil {
		return 0
	}
	return uint64(C.snarf_size_bits(f.ptr))
}
