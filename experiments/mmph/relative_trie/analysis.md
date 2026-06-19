# PSig / Memory Study Summary

## Inputs
- `mmph/relative_trie/study/data/grid_main_v2.csv`
- `mmph/relative_trie/study/data/grid_focus_v2.csv`
- `mmph/relative_trie/study/data/grid_focus_extra_s16.csv`
- `mmph/relative_trie/study/data/grid_focus_extra_s8.csv`
- `mmph/relative_trie/study/data/grid_focus_extra_s32.csv`
- `mmph/relative_trie/study/data/grid_focus_big_s32.csv`
- `mmph/relative_trie/study/data/grid_focus_small_s8.csv`
- `/Users/andrei.ogurtsov/Thesis/mmph/relative_trie/study/memory_bench_v2.txt`

## Main Findings
- On `grid_main_v2` (144 scenarios, 64 trials each): mean success by S is S=8: 0.166, S=16: 0.893, S=32: 1.000.
- `S=32` is fully stable in this grid (all scenarios succeeded in all trials); S=8 fails for most medium/large settings.
- `S=16` is mixed: stable up to moderate `n`, but for `n=131072` several `w` values show severe degradation.
- This confirms that theorem-based `S` from per-query bound (`epsilon_query = m/n`) is necessary but not sufficient for high probability of full-structure build success.

## Plots
- [S=8 empirical](study/plots/s8_success_vs_n.svg), [S=8 theory](study/plots/s8_success_vs_n_theory.svg), [S=8 overlay](study/plots/s8_success_vs_n_overlay.svg)
- [S=16 empirical](study/plots/s16_success_vs_n.svg), [S=16 theory](study/plots/s16_success_vs_n_theory.svg), [S=16 overlay](study/plots/s16_success_vs_n_overlay.svg)
- [S=32 empirical](study/plots/s32_success_vs_n.svg), [S=32 theory](study/plots/s32_success_vs_n_theory.svg), [S=32 overlay](study/plots/s32_success_vs_n_overlay.svg)
- [Grid size report (bpk vs n)](study/plots/grid_size_report_bpk_vs_n.svg)
- [Memory bits/key at 32768 keys](study/plots/memory_bits_per_key_keys32768.svg)

## Detailed Theory: how build-success probability was computed
- References used: `papers/MMPH/Definitions-and-Tools.md`, `papers/MMPH/Section-3-Bucketing.md`, `papers/MMPH/Section-4-Relative-Ranking.md` (Theorem 4.1), `papers/MMPH/Section-5-Relative-Trie.md` (Theorem 5.2).
- Goal of this section: derive (i) required PSig width `S` from the paper, and (ii) build-success probability for our concrete implementation.

### 0. Notation aligned with the paper
- `n = |S|`: number of keys for which queries must be correct.
- `m = |D|`: number of delimiters (one per bucket). For bucket size `b`, typically `m = ceil(n/b)`.
- `w`: max key length in bits.
- `k`: number of signature checks during fat binary search; by Theorem 4.1 analysis, `k <= ceil(log2(w))`.
- `S`: PSig width in bits (hash/signature length stored in trie entries).
- `R`: max rebuild attempts (`maxTrieRebuilds = 100` in current code).

### 1. From Theorem 4.1 to per-query failure
- Theorem 4.1 states that each signature check uses `log2(log2(w)) + log2(1/epsilon_query)` bits.
- Therefore with fixed width `S`, one comparison false-match probability is:
$$
p_{\mathrm{cmp}} = 2^{-S}
$$
- Query performs up to `k` checks. Union bound used in the theorem proof:
$$
p_{\mathrm{query\_fail}} \leq k\cdot 2^{-S}
$$
- In code/plots we also use tighter independent-check approximation:
$$
p_{\mathrm{query\_fail}} \approx 1-(1-2^{-S})^{k}
$$
- This is implemented by `theory_query_false_positive()` with `k = max(1, ceil(log2(max(2, w))))`.

### 2. Why `epsilon_query = m/n` in Theorem 5.2
- Theorem 5.2 sets per-query error target to `epsilon_query = m/n`.
- Then expected number of misclassified keys over all `n` keys is:
$$
\mathbb{E}[|E|] = n\cdot \varepsilon_{\mathrm{query}} = m
$$
- Paper then stores explicit corrections for this set `E` (relative-membership + stored exact answers), yielding exact queries on `S`.
- Substituting `epsilon_query = m/n` into Theorem 4.1 gives required PSig width:
$$
S \geq \log_2\!\log_2(w) + \log_2\!\left(\frac{n}{m}\right)
$$
- For fixed bucket size `b` where `m \approx n/b`, this becomes:
$$
S \geq \log_2\!\log_2(w) + \log_2(b)
$$
- For `b=256`, this is roughly `\log_2\!\log_2(w) + 8` (plus integer ceiling/constant slack).

### 3. Mapping paper guarantees to our builder
- Current implementation does **not** store correction set `E` from Theorem 5.2.
- Instead, one build attempt is accepted only if `validateAllKeys` succeeds for all keys (`n` checks pass).
- So we model probability of *zero failures in one attempt*.

### 4. One-attempt success model
- Let `p = p_query_fail` from Step 1.
- Independent-query approximation gives:
$$
p_{\mathrm{attempt\_success}} \approx (1-p)^n
$$
- For small `p`, this is close to Poisson form:
$$
p_{\mathrm{attempt\_success}} \approx e^{-\lambda},\quad \lambda = n\cdot p
$$
- This explains sharp phase transitions: once `n * p` is not small, all-keys success becomes unlikely.

### 5. Rebuild attempts
- Builder retries with fresh seeds up to `R=100` attempts.
- With approximate independence between attempts:
$$
p_{\mathrm{build\_success}} \approx 1-(1-p_{\mathrm{attempt\_success}})^R
$$
- This is exactly what `theory_build_success()` computes and what is plotted in `*_theory.svg` and `*_overlay.svg`.

### 6. Conservative bound vs approximation
- Conservative theorem-style upper bound for query failure:
$$
p_{\mathrm{query\_fail}}^{\mathrm{bound}} = \min\left(1, k\cdot2^{-S}\right)
$$
- This yields a lower bound for one-attempt success:
$$
p_{\mathrm{attempt\_success}} \ge (1-p_{\mathrm{query\_fail}}^{\mathrm{bound}})^n
$$
- We plot the independent approximation because it tracks observed trends better than the loose union-bound lower bound.

### 7. Why empirical and theory differ
- Hash/signature events are not perfectly independent across keys (shared trie paths).
- `validateAllKeys` behavior depends on actual trie shape, delimiter distribution, and candidate selection details (`LowerBound`).
- Attempt-to-attempt outcomes may also be correlated for some key sets despite seed changes.
- The implementation applies a length-consistency collision filter in `GetExistingPrefix` that rejects MPH hits with `extentLen < fFast`, removing a class of deterministic false positives not modeled in the paper. See `zfasttrie/getexistingprefix_collision_filter.md`.
- Therefore we expect qualitative trend alignment, not exact numeric equality.

### Worked example (`n=131072, w=128, S=16`)
- `k = 7`
$$
p_{\mathrm{query\_fail}} \approx 0.00010680663
$$
$$
\lambda = n\cdot p_{\mathrm{query\_fail}} \approx 13.999359
$$
$$
p_{\mathrm{attempt\_success}} \approx 8.3143991e-07
$$
$$
p_{\mathrm{build\_success}}(\mathrm{theory}, R=100) \approx 8.3140569e-05
$$
- `p_build_success(empirical from focus grid) = 0.28125`

## Margin Analysis
- `summary_by_margin.csv` shows behavior grouped by `s_margin_bits = S - S_required`.
- Positive margin improves success rate but does not guarantee `~1.0` success for largest `n`.

## Memory Snapshot (from `mmph/relative_trie/study/memory_bench_v2.txt`)
- Stable regime (`keys>=8192`): RLOC avg=54.33 bits/key, LERLOC avg=182.43 bits/key.
- MMPH baseline from paper chart: ~14 bits/key (for large n).

## Generated Artifacts
- [`data/summary_by_s.csv`](study/data/summary_by_s.csv)
- [`data/summary_by_margin.csv`](study/data/summary_by_margin.csv)
- [`data/worst_cases.csv`](study/data/worst_cases.csv)
- [`data/memory_points.csv`](study/data/memory_points.csv)
- [`data/theory_focus_success.csv`](study/data/theory_focus_success.csv)
- [`plots/success_rate_by_s.svg`](study/plots/success_rate_by_s.svg)
- [`plots/s16_success_vs_n.svg`](study/plots/s16_success_vs_n.svg)
- [`plots/s16_success_vs_n_theory.svg`](study/plots/s16_success_vs_n_theory.svg)
- [`plots/s16_success_vs_n_overlay.svg`](study/plots/s16_success_vs_n_overlay.svg)
- [`plots/s8_success_vs_n.svg`](study/plots/s8_success_vs_n.svg)
- [`plots/s8_success_vs_n_theory.svg`](study/plots/s8_success_vs_n_theory.svg)
- [`plots/s8_success_vs_n_overlay.svg`](study/plots/s8_success_vs_n_overlay.svg)
- [`plots/s32_success_vs_n.svg`](study/plots/s32_success_vs_n.svg)
- [`plots/s32_success_vs_n_theory.svg`](study/plots/s32_success_vs_n_theory.svg)
- [`plots/s32_success_vs_n_overlay.svg`](study/plots/s32_success_vs_n_overlay.svg)
- [`plots/grid_size_report_bpk_vs_n.svg`](study/plots/grid_size_report_bpk_vs_n.svg)
- [`plots/memory_bits_per_key_keys32768.svg`](study/plots/memory_bits_per_key_keys32768.svg)
