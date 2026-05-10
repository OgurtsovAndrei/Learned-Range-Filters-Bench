package datasets_test

import (
	"path/filepath"
	"runtime"
	"testing"

	"Thesis-bench-industry/bench/datasets"
)

func sosdPath(name string) string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), "..", "sosd_data", name)
}

func TestSOSDReader_FB(t *testing.T) {
	r := &datasets.SOSDReader{
		Path:    sosdPath("fb_200M_uint64"),
		Label:   "sosd_fb",
		KeyType: datasets.SOSDUint64,
		MaxKeys: 1_000_000,
	}
	keys, err := r.Keys()
	if err != nil {
		t.Skipf("dataset not available: %v", err)
	}
	if len(keys) != 1_000_000 {
		t.Errorf("MaxKeys=1M ignored: got %d", len(keys))
	}
	for i := 1; i < len(keys); i++ {
		if keys[i] <= keys[i-1] {
			t.Fatalf("not sorted-unique at i=%d: %d <= %d", i, keys[i], keys[i-1])
		}
	}
	if r.Name() != "sosd_fb" {
		t.Errorf("Name() = %q, want %q", r.Name(), "sosd_fb")
	}
}

func TestSOSDReader_Books32(t *testing.T) {
	r := &datasets.SOSDReader{
		Path:    sosdPath("books_200M_uint32"),
		Label:   "sosd_books",
		KeyType: datasets.SOSDUint32,
		MaxKeys: 100_000,
	}
	keys, err := r.Keys()
	if err != nil {
		t.Skipf("dataset not available: %v", err)
	}
	if len(keys) == 0 {
		t.Fatal("no keys")
	}
	for i := 1; i < len(keys); i++ {
		if keys[i] <= keys[i-1] {
			t.Fatalf("not sorted-unique at i=%d", i)
		}
	}
}

func TestSOSDReader_Books800(t *testing.T) {
	r := &datasets.SOSDReader{
		Path:    sosdPath("books_800M_uint64"),
		Label:   "sosd_books_800M",
		KeyType: datasets.SOSDUint64,
		MaxKeys: 1_000_000,
	}
	keys, err := r.Keys()
	if err != nil {
		t.Skipf("dataset not available (run sosd_data/download.sh): %v", err)
	}
	if len(keys) != 1_000_000 {
		t.Errorf("MaxKeys=1M cap ignored: got %d", len(keys))
	}
	for i := 1; i < len(keys); i++ {
		if keys[i] <= keys[i-1] {
			t.Fatalf("not sorted-unique at i=%d", i)
		}
	}
}
