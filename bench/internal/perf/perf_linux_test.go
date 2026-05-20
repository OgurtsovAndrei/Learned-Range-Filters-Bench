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
