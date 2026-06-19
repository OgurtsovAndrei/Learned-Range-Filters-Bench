package main

import (
	"Thesis/bits"
	bucket "Thesis-bench-industry/experiments/mmph/relative_trie"
	"Thesis-bench-industry/experiments/mmph/paramselect"
	"Thesis-bench-industry/experiments/trie/zft"
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type scenario struct {
	n     int
	wBits int
	sBits int
}

type scenarioResult struct {
	scenario
	bucketSize          int
	m                   int
	eBits               int
	iBits               int
	checks              int
	sRequiredBits       int
	sMarginBits         int
	trials              int
	successes           int64
	failures            int64
	avgAttemptsSuccess  float64
	p50AttemptsSuccess  float64
	p95AttemptsSuccess  float64
	maxAttemptsSuccess  int64
	avgSizeBytesSuccess float64
	avgSizeBitsSuccess  float64
	avgBPKSuccess       float64
	epsilonTargetQuery  float64
	theoryQueryFpByS    float64
}

func main() {
	var (
		outPath    = flag.String("out", "mmph/relative_trie/study/data/results.csv", "Output CSV path")
		nsArg      = flag.String("n", "1024,8192,32768", "Comma-separated key counts")
		wsArg      = flag.String("w", "64,128,256,512,1024", "Comma-separated key lengths in bits (prefer multiples of 8)")
		sArg       = flag.String("s", "8,16,32", "Comma-separated PSig widths in bits")
		bucketSize = flag.Int("bucket", 256, "Bucket size")
		trials     = flag.Int("trials", 48, "Trials per scenario")
		workers    = flag.Int("workers", runtime.NumCPU(), "Parallel workers per scenario")
		seed       = flag.Int64("seed", time.Now().UnixNano(), "Base RNG seed")
	)
	flag.Parse()

	ns := parseCSVInts(*nsArg)
	ws := parseCSVInts(*wsArg)
	ss := parseCSVInts(*sArg)
	if len(ns) == 0 || len(ws) == 0 || len(ss) == 0 {
		fail("n, w and s must be non-empty")
	}
	if *bucketSize <= 0 {
		fail("bucket must be > 0")
	}
	if *trials <= 0 {
		fail("trials must be > 0")
	}
	if *workers <= 0 {
		fail("workers must be > 0")
	}

	runtime.GOMAXPROCS(*workers)

	scenarios := make([]scenario, 0, len(ns)*len(ws)*len(ss))
	for _, n := range ns {
		for _, w := range ws {
			for _, s := range ss {
				scenarios = append(scenarios, scenario{n: n, wBits: w, sBits: s})
			}
		}
	}

	if err := os.MkdirAll(dirOf(*outPath), 0o755); err != nil {
		fail("failed to create output directory: %v", err)
	}

	f, err := os.Create(*outPath)
	if err != nil {
		fail("failed to create output file: %v", err)
	}
	defer f.Close()

	wr := csv.NewWriter(f)
	defer wr.Flush()

	header := []string{
		"n", "w_bits", "s_bits", "bucket_size", "m_delimiters",
		"e_bits", "i_bits", "s_required_bits",
		"s_margin_bits", "checks",
		"trials", "successes", "failures", "fail_rate",
		"success_rate", "avg_attempts_success", "p50_attempts_success",
		"p95_attempts_success", "max_attempts_success",
		"avg_size_bytes_success", "avg_size_bits_success", "bpk",
		"epsilon_target_query", "theory_query_fp_by_s",
	}
	mustWrite(wr, header)

	for idx, sc := range scenarios {
		fmt.Printf("[%d/%d] n=%d w=%d s=%d ...\n", idx+1, len(scenarios), sc.n, sc.wBits, sc.sBits)
		res := runScenario(sc, *bucketSize, *trials, *workers, *seed+int64(idx)*1_000_003)
		sizeBytesStr := formatFloatOrNone(res.avgSizeBytesSuccess, res.successes)
		sizeBitsStr := formatFloatOrNone(res.avgSizeBitsSuccess, res.successes)
		bpkStr := formatFloatOrNone(res.avgBPKSuccess, res.successes)
		row := []string{
			itoa(res.n),
			itoa(res.wBits),
			itoa(res.sBits),
			itoa(res.bucketSize),
			itoa(res.m),
			itoa(res.eBits),
			itoa(res.iBits),
			itoa(res.sRequiredBits),
			itoa(res.sMarginBits),
			itoa(res.checks),
			itoa(res.trials),
			i64toa(res.successes),
			i64toa(res.failures),
			fmt.Sprintf("%.8f", float64(res.failures)/float64(res.trials)),
			fmt.Sprintf("%.8f", float64(res.successes)/float64(res.trials)),
			fmt.Sprintf("%.6f", res.avgAttemptsSuccess),
			fmt.Sprintf("%.2f", res.p50AttemptsSuccess),
			fmt.Sprintf("%.2f", res.p95AttemptsSuccess),
			i64toa(res.maxAttemptsSuccess),
			sizeBytesStr,
			sizeBitsStr,
			bpkStr,
			fmt.Sprintf("%.8f", res.epsilonTargetQuery),
			fmt.Sprintf("%.8f", res.theoryQueryFpByS),
		}
		mustWrite(wr, row)
		wr.Flush()
	}

	fmt.Printf("done: %s\n", *outPath)
}

func runScenario(sc scenario, bucketSize int, trials int, workers int, baseSeed int64) scenarioResult {
	m := paramselect.BucketCount(sc.n, bucketSize)
	eBits := paramselect.WidthForBitLength(sc.wBits)
	iBits := paramselect.WidthForDelimiterTrieIndex(m)
	sRequired := paramselect.SignatureBitsRelativeTrie(sc.wBits, sc.n, m)
	epsilon := float64(m) / float64(sc.n)
	checks := max(1, int(math.Ceil(math.Log2(float64(max(2, sc.wBits))))))
	pQueryByS := 1.0 - math.Pow(1.0-math.Pow(2, -float64(sc.sBits)), float64(checks))

	jobs := make(chan int, trials)
	var wg sync.WaitGroup

	var successes int64
	var failures int64
	var attemptsSum int64
	var sizeBytesSum int64
	var maxAttempts int64
	successAttempts := make([]int64, 0, trials)
	var attemptsMu sync.Mutex

	workerCount := min(workers, trials)
	for w := 0; w < workerCount; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for t := range jobs {
				trialSeed := mixSeed(baseSeed, int64(workerID), int64(t))
				keys := genUniqueSortedKeys(sc.n, sc.wBits, trialSeed)
				attempts, sizeBytes, err := buildByWidths(keys, eBits, sc.sBits, iBits, uint64(trialSeed))
				if err != nil {
					atomic.AddInt64(&failures, 1)
					continue
				}
				atomic.AddInt64(&successes, 1)
				at := int64(attempts)
				atomic.AddInt64(&attemptsSum, at)
				atomic.AddInt64(&sizeBytesSum, int64(sizeBytes))
				updateMaxInt64(&maxAttempts, at)
				attemptsMu.Lock()
				successAttempts = append(successAttempts, at)
				attemptsMu.Unlock()
			}
		}(w)
	}

	for t := 0; t < trials; t++ {
		jobs <- t
	}
	close(jobs)
	wg.Wait()

	avgAttempts := 0.0
	avgSizeBytes := 0.0
	avgSizeBits := 0.0
	avgBPK := 0.0
	if successes > 0 {
		avgAttempts = float64(attemptsSum) / float64(successes)
		avgSizeBytes = float64(sizeBytesSum) / float64(successes)
		avgSizeBits = avgSizeBytes * 8.0
		avgBPK = avgSizeBits / float64(sc.n)
	}
	p50Attempts := quantileInt64(successAttempts, 0.50)
	p95Attempts := quantileInt64(successAttempts, 0.95)

	return scenarioResult{
		scenario:            sc,
		bucketSize:          bucketSize,
		m:                   m,
		eBits:               eBits,
		iBits:               iBits,
		checks:              checks,
		sRequiredBits:       sRequired,
		sMarginBits:         sc.sBits - sRequired,
		trials:              trials,
		successes:           successes,
		failures:            failures,
		avgAttemptsSuccess:  avgAttempts,
		p50AttemptsSuccess:  p50Attempts,
		p95AttemptsSuccess:  p95Attempts,
		maxAttemptsSuccess:  maxAttempts,
		avgSizeBytesSuccess: avgSizeBytes,
		avgSizeBitsSuccess:  avgSizeBits,
		avgBPKSuccess:       avgBPK,
		epsilonTargetQuery:  epsilon,
		theoryQueryFpByS:    pQueryByS,
	}
}

func buildByWidths(keys []bits.BitString, eBits int, sBits int, iBits int, seed uint64) (int, int, error) {
	switch eBits {
	case 8:
		return buildByWidthsE[uint8](keys, sBits, iBits, seed)
	case 16:
		return buildByWidthsE[uint16](keys, sBits, iBits, seed)
	case 32:
		return buildByWidthsE[uint32](keys, sBits, iBits, seed)
	default:
		return 0, 0, fmt.Errorf("unsupported eBits=%d", eBits)
	}
}

func buildByWidthsE[E zft.UNumber](keys []bits.BitString, sBits int, iBits int, seed uint64) (int, int, error) {
	switch sBits {
	case 8:
		return buildByWidthsES[E, uint8](keys, iBits, seed)
	case 16:
		return buildByWidthsES[E, uint16](keys, iBits, seed)
	case 32:
		return buildByWidthsES[E, uint32](keys, iBits, seed)
	}
	return 0, 0, fmt.Errorf("unsupported sBits=%d", sBits)
}

func buildByWidthsES[E zft.UNumber, S zft.UNumber](keys []bits.BitString, iBits int, seed uint64) (int, int, error) {
	switch iBits {
	case 8:
		mh, err := bucket.NewMonotoneHashWithTrieSeeded[E, S, uint8](keys, seed)
		if err != nil {
			return 0, 0, err
		}
		return mh.TrieRebuildAttempts, mh.ByteSize(), nil
	case 16:
		mh, err := bucket.NewMonotoneHashWithTrieSeeded[E, S, uint16](keys, seed)
		if err != nil {
			return 0, 0, err
		}
		return mh.TrieRebuildAttempts, mh.ByteSize(), nil
	case 32:
		mh, err := bucket.NewMonotoneHashWithTrieSeeded[E, S, uint32](keys, seed)
		if err != nil {
			return 0, 0, err
		}
		return mh.TrieRebuildAttempts, mh.ByteSize(), nil
	default:
		return 0, 0, fmt.Errorf("unsupported iBits=%d", iBits)
	}
}

func genUniqueSortedKeys(n int, wBits int, seed int64) []bits.BitString {
	rng := rand.New(rand.NewSource(seed))
	byteLen := (wBits + 7) / 8
	keys := make([]bits.BitString, 0, n)
	seen := make(map[string]struct{}, n)

	for len(keys) < n {
		buf := make([]byte, byteLen)
		for i := range buf {
			buf[i] = byte(rng.Intn(256))
		}
		// Mask tail bits when wBits is not byte-aligned.
		if rem := wBits % 8; rem != 0 {
			mask := byte((1 << uint(rem)) - 1)
			buf[byteLen-1] &= mask
		}
		s := string(buf)
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		keys = append(keys, bits.NewFromText(s))
	}

	sort.Slice(keys, func(i, j int) bool {
		return keys[i].Compare(keys[j]) < 0
	})
	return keys
}

func parseCSVInts(v string) []int {
	if strings.TrimSpace(v) == "" {
		return nil
	}
	parts := strings.Split(v, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil {
			fail("failed to parse int %q: %v", p, err)
		}
		out = append(out, n)
	}
	return out
}

func mustWrite(w *csv.Writer, row []string) {
	if err := w.Write(row); err != nil {
		fail("failed to write csv row: %v", err)
	}
}

func dirOf(path string) string {
	i := strings.LastIndex(path, "/")
	if i < 0 {
		return "."
	}
	if i == 0 {
		return "/"
	}
	return path[:i]
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

func itoa(v int) string {
	return strconv.Itoa(v)
}

func i64toa(v int64) string {
	return strconv.FormatInt(v, 10)
}

func formatFloatOrNone(v float64, successes int64) string {
	if successes == 0 {
		return "none"
	}
	return fmt.Sprintf("%.6f", v)
}

func quantileInt64(values []int64, q float64) float64 {
	if len(values) == 0 {
		return 0
	}
	if q <= 0 {
		return float64(minInt64(values))
	}
	if q >= 1 {
		return float64(maxInt64(values))
	}
	cp := append([]int64(nil), values...)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	pos := int(math.Round(q * float64(len(cp)-1)))
	if pos < 0 {
		pos = 0
	}
	if pos >= len(cp) {
		pos = len(cp) - 1
	}
	return float64(cp[pos])
}

func minInt64(values []int64) int64 {
	m := values[0]
	for _, v := range values[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func maxInt64(values []int64) int64 {
	m := values[0]
	for _, v := range values[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func mixSeed(base int64, a int64, b int64) int64 {
	x := uint64(base) + 0x9e3779b97f4a7c15
	x ^= uint64(a) + 0x9e3779b97f4a7c15 + (x << 6) + (x >> 2)
	x ^= uint64(b) + 0x9e3779b97f4a7c15 + (x << 6) + (x >> 2)
	return int64(x)
}

func updateMaxInt64(dst *int64, value int64) {
	for {
		cur := atomic.LoadInt64(dst)
		if value <= cur {
			return
		}
		if atomic.CompareAndSwapInt64(dst, cur, value) {
			return
		}
	}
}
