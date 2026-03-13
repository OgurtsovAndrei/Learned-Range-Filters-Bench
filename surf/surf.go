package surf

/*
#cgo CXXFLAGS: -std=c++17
#cgo LDFLAGS: -L${SRCDIR}/build -lsurf_wrapper -lc++ -lm
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

type SuffixType int

const (
	SuffixNone SuffixType = 0
	SuffixHash SuffixType = 1
	SuffixReal SuffixType = 2
)

type SuRFFilter struct {
	ptr C.SuRFPtr
	n   int
}

func New(keys []uint64, st SuffixType, hashBits, realBits int) *SuRFFilter {
	if len(keys) == 0 {
		return &SuRFFilter{}
	}
	ptr := C.surf_new(
		(*C.uint64_t)(unsafe.Pointer(&keys[0])),
		C.size_t(len(keys)),
		C.int(st),
		C.int(hashBits),
		C.int(realBits),
	)
	f := &SuRFFilter{ptr: ptr, n: len(keys)}
	runtime.SetFinalizer(f, func(obj *SuRFFilter) {
		if obj.ptr != nil {
			C.surf_free(obj.ptr)
			obj.ptr = nil
		}
	})
	return f
}

// IsEmpty returns true if the range [lo, hi] is definitely empty.
func (f *SuRFFilter) IsEmpty(lo, hi uint64) bool {
	if f.ptr == nil {
		return true
	}
	return C.surf_query(f.ptr, C.uint64_t(lo), C.uint64_t(hi)) == 0
}

func (f *SuRFFilter) SizeInBits() uint64 {
	if f.ptr == nil {
		return 0
	}
	return uint64(C.surf_size_bits(f.ptr))
}
