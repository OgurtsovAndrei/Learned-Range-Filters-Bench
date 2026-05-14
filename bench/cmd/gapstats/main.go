package main

import (
	"fmt"
	"math"
	"sort"
	"Thesis-bench-industry/bench/internal/keygen"
)

func gapStats(keys []uint64) (min, p10, p50, p90, max, mean float64) {
	gaps := make([]float64, len(keys)-1)
	sum := 0.0
	for i := 1; i < len(keys); i++ {
		g := float64(keys[i] - keys[i-1])
		gaps[i-1] = g
		sum += g
	}
	sort.Float64s(gaps)
	n := len(gaps)
	pct := func(p float64) float64 {
		idx := p / 100.0 * float64(n-1)
		lo := int(math.Floor(idx))
		hi := int(math.Ceil(idx))
		if lo == hi { return gaps[lo] }
		return gaps[lo]*(float64(hi)-idx) + gaps[hi]*(idx-float64(lo))
	}
	return gaps[0], pct(10), pct(50), pct(90), gaps[n-1], sum / float64(n)
}

func bits(v uint64) float64 {
	if v == 0 { return 0 }
	return math.Log2(float64(v)) + 1
}

func main() {
	type ds struct {
		name string
		keys []uint64
		err  error
	}
	datasets := []ds{
		{"FB (uint64)",   nil, nil},
		{"Wiki (uint64)", nil, nil},
		{"OSM (uint64)",  nil, nil},
		{"Books (uint32)",nil, nil},
	}
	var err error
	datasets[0].keys, err = keygen.LoadSOSDUint64(keygen.SOSDPath("fb_200M_uint64"), 0)
	datasets[0].err = err
	datasets[1].keys, err = keygen.LoadSOSDUint64(keygen.SOSDPath("wiki_ts_200M_uint64"), 0)
	datasets[1].err = err
	datasets[2].keys, err = keygen.LoadSOSDUint64(keygen.SOSDPath("osm_cellids_800M_uint64"), 0)
	datasets[2].err = err
	datasets[3].keys, err = keygen.LoadSOSDUint32(keygen.SOSDPath("books_200M_uint32"), 0)
	datasets[3].err = err

	fmt.Printf("%-18s  %12s  %8s  %8s  %8s  %8s  %8s  %8s  %12s\n",
		"Dataset", "n", "U bits", "min gap", "P10", "P50", "P90", "mean", "max gap")
	fmt.Println("----------------------------------------------------------------------------------------------------------------------")
	for _, d := range datasets {
		if d.err != nil {
			fmt.Printf("%-18s  ERROR: %v\n", d.name, d.err)
			continue
		}
		n := len(d.keys)
		universe := d.keys[n-1] - d.keys[0]
		uBits := bits(universe)
		mn, p10, p50, p90, mx, mean := gapStats(d.keys)
		fmt.Printf("%-18s  %12d  %8.1f  %8.0f  %8.0f  %8.0f  %8.0f  %8.1f  %12.0f\n",
			d.name, n, uBits, mn, p10, p50, p90, mean, mx)
	}
}
