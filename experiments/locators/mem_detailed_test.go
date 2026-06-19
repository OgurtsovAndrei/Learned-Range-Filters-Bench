package locators

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/locators/lerloc"
	"sort"
	"testing"
)

func TestMemDetailed(t *testing.T) {
	keys := []bits.BitString{
		bits.NewFromText("apple"),
		bits.NewFromText("apply"),
		bits.NewFromText("ball"),
		bits.NewFromText("bat"),
	}

	sort.Slice(keys, func(i, j int) bool {
		return keys[i].Compare(keys[j]) < 0
	})

	lerl, err := lerloc.NewLocalExactRangeLocator(keys)
	if err != nil {
		t.Fatalf("Failed to build LERLOC: %v", err)
	}

	report := lerl.MemDetailed()
	
	// Verify it has total bytes
	if report.TotalBytes == 0 {
		t.Errorf("Expected non-zero total bytes in report")
	}

	// Verify JSON output
	jsonStr := report.JSON()
	if len(jsonStr) < 10 {
		t.Errorf("JSON report too short: %s", jsonStr)
	}

	// Print for manual inspection
	t.Logf("Hierarchical Memory Report:\n%s", report.String())
	t.Logf("JSON Memory Report: %s", jsonStr)
}
