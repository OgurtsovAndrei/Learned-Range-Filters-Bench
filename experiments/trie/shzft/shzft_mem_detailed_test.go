package shzft

import (
	"Thesis/testutils"
	"fmt"
	"testing"
)

func TestSHZFT_MemoryBreakdown_Pure(t *testing.T) {
	n := 32768
	l := 64

	keys := testutils.GetBenchKeys(l, n)

	shzft := NewSuccinctHZFastTrie(keys)

	report := shzft.MemDetailed()
	
	fmt.Printf("\n=== Pure N Memory Analysis ===\n")
	fmt.Printf("N = %d, L = %d\n", n, l)
	fmt.Printf("Total Bytes: %d\n", report.TotalBytes)
	fmt.Printf("Total Bits / Key: %.2f\n", float64(report.TotalBytes*8)/float64(n))

	for _, child := range report.Children {
		fmt.Printf("  - %s: %d bytes (%.2f bits/key)\n", child.Name, child.TotalBytes, float64(child.TotalBytes*8)/float64(n))
	}
	
	fmt.Printf("\nEntries stats:\n")
	fmt.Printf("Total entries (BV length) = %d (%.2f entries/key)\n", shzft.GetNumEntries(), float64(shzft.GetNumEntries())/float64(n))
	fmt.Printf("True descriptors          = %d\n", shzft.GetTrueEntries())
	fmt.Printf("Pseudo descriptors        = %d\n", shzft.GetNumEntries() - shzft.GetTrueEntries())
	fmt.Printf("deltaBits                 = %d\n", shzft.deltaBits)
	fmt.Printf("==============================\n\n")
}
