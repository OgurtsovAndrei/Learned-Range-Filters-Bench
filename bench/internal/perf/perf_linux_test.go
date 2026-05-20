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
