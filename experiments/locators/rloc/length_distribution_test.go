package rloc

import (
	"Thesis-bench-industry/experiments/trie/zft"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestExportLengthDistribution(t *testing.T) {
	lengthsToTest := []int{64, 256}
	keyCount := 100000

	type distributionData struct {
		L           int            `json:"L"`
		N           int            `json:"N"`
		PSize       int            `json:"P_size"`
		Frequencies map[int]int    `json:"frequencies"` // bit length -> count
	}

	var allData []distributionData

	for _, L := range lengthsToTest {
		t.Logf("Generating %d keys of length %d...", keyCount, L)
		keys := GetBenchKeys(L, keyCount)

		t.Log("Building ZFastTrie...")
		zt := zft.Build(keys)

		t.Log("Collecting P sorted items...")
		sortedItems, _ := collectPSortedItems(zt)

		frequencies := make(map[int]int)
		for _, item := range sortedItems {
			size := int(item.bs.Size())
			frequencies[size]++
		}

		allData = append(allData, distributionData{
			L:           L,
			N:           keyCount,
			PSize:       len(sortedItems),
			Frequencies: frequencies,
		})
	}

	outDir := "../benchmarks/raw"
	if err := os.MkdirAll(outDir, 0755); err != nil {
		t.Fatalf("Failed to create output directory: %v", err)
	}

	outPath := filepath.Join(outDir, "p_length_distribution.json")
	file, err := os.Create(outPath)
	if err != nil {
		t.Fatalf("Failed to create file: %v", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(allData); err != nil {
		t.Fatalf("Failed to encode JSON: %v", err)
	}

	t.Logf("Successfully wrote distribution to %s", outPath)
}