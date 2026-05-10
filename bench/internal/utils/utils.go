package utils

import (
	"Thesis/testutils"
)

func NormalizedCDF(keys []uint64, sampleEvery int) []testutils.Point {
	n := len(keys)
	if n < 2 {
		return nil
	}
	minK, maxK := float64(keys[0]), float64(keys[n-1])
	span := maxK - minK
	if span == 0 {
		span = 1
	}
	pts := make([]testutils.Point, 0, n/sampleEvery+2)
	pts = append(pts, testutils.Point{X: 0, Y: 0})
	for i := 0; i < n; i += sampleEvery {
		x := (float64(keys[i]) - minK) / span
		y := float64(i+1) / float64(n)
		pts = append(pts, testutils.Point{X: x, Y: y})
	}
	pts = append(pts, testutils.Point{X: 1, Y: 1})
	return pts
}

func Histogram(keys []uint64, nBins int) []testutils.Point {
	n := len(keys)
	if n < 2 {
		return nil
	}
	minK, maxK := float64(keys[0]), float64(keys[n-1])
	span := maxK - minK
	if span == 0 {
		span = 1
	}
	counts := make([]int, nBins)
	for _, k := range keys {
		bin := int((float64(k) - minK) / span * float64(nBins))
		if bin >= nBins {
			bin = nBins - 1
		}
		counts[bin]++
	}
	maxCount := 0
	for _, c := range counts {
		if c > maxCount {
			maxCount = c
		}
	}
	pts := make([]testutils.Point, 0, nBins)
	for i, c := range counts {
		pts = append(pts, testutils.Point{
			X: float64(i) / float64(nBins),
			Y: float64(c) / float64(maxCount),
		})
	}
	return pts
}
