// Package go_boomphf_bs is a wrapper for the optimized BBHash implementation
package go_boomphf_bs

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/mmph/go-boomphf-bs/inline"
)

// H is an alias to the most efficient implementation (inline)
type H = inline.H

// Gamma is the recommended default value for controlling space vs. construction speed
const Gamma = 2.0

// NewDefault constructs a perfect hash function with the default gamma value (2.0)
func NewDefault(keys []bits.BitString) *H {
	return inline.New(Gamma, keys)
}

// New preserves the original signature for compatibility, using the inline implementation
func New(gamma float64, keys []bits.BitString) *H {
	return inline.New(gamma, keys)
}
