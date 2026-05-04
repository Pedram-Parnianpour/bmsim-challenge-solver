# Audit Report — BMsim Challenge Compliance

Full verification of our solver against every requirement in
https://github.com/pulseq-cest/BMsim_challenge

Last verified: May 2026. All checks passed programmatically.

---

## General settings (from challenge README)

| # | Requirement | Our Implementation | Status |
|---|---|---|---|
| 1 | γ = 42.5764 MHz/T (NIST rounded) | Read from YAML: 267.5154109126009 rad/(s·µT) ÷ 2π = 42.5764 MHz/T | ✅ |
| 2 | Pool fraction def 1: water f=1 | water f=1.0 in all YAMLs; k_ac = k × f_pool / f_water | ✅ |
| 3 | Fully relaxed Zi=1 per offset | reset_init_mag=True, scale=1.0; Mz_water=1.0 verified | ✅ |
| 4 | Post-prep delay = 6.5 ms | Cases 1–4: hardcoded `post_sat_delay=0.0065`; Cases 5–8: gz_id block in seq file = 6.5 ms | ✅ |
| 5 | Normalization scan at −300 ppm | Cases 1–4: M0 scan forced to −300 ppm in code; Cases 5–8: M0_offset=−300 in seq file | ✅ |
| 6 | MT pool: full 3-component [Mx,My,Mz] | `add_pool()` gives all 3 components to every pool including MT | ✅ |
| 7 | Larmor = 127.7292 MHz at 3T | 42.5764 MHz/T × 3T = 127.7292 MHz | ✅ |

---

## Per-case verification

### Cases 1–4 (CW block pulses)

| Case | tp | B1 | n_pulses | Z-offsets | M0 row | Total rows | Status |
|---|---|---|---|---|---|---|---|
| 1 | 15 s | 2.0 µT | 1 | 301 (−15:0.1:15 ppm) | −300 ppm | 302 | ✅ |
| 2 | 2 s | 2.0 µT | 1 | 301 (−15:0.1:15 ppm) | −300 ppm | 302 | ✅ |
| 3 | 2 s | 2.0 µT | 1 | 301 (−15:0.1:15 ppm) | −300 ppm | 302 | ✅ |
| 4 | 5 ms | 3.7 µT | 1 | 81 (−2:0.05:2 ppm) | −300 ppm | 82 | ✅ |

**Note on cases 1–3 M0 offset:** The seq files for cases 1–3 specify M0_offset = −1560 ppm
internally. Our code overrides this and always simulates the M0 scan at −300 ppm to match
the convention used by all other groups. Verified: our value at −300 ppm for case 1
(0.9999925651) matches other groups' values (0.9999925650) to 10 decimal places.

### Cases 5–8 (shaped pulses and pulse trains)

| Case | Shape | tp | n_pulses | td | B1 | Z-offsets | M0 row | max_samples | Total rows | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | Gaussian | 50 ms | 1 | — | 1.9962 µT rms | 201 (−2:0.02:2 ppm) | −300 ppm | 200 | 202 | ✅ |
| 6 | Gaussian | 50 ms | 36 | 5 ms | 1.9962 µT rms | 301 (−15:0.1:15 ppm) | −300 ppm | 200 | 302 | ✅ |
| 7 | Gaussian | 50 ms | 36 | 5 ms | 1.9962 µT rms | 301 (−15:0.1:15 ppm) | −300 ppm | 200 | 302 | ✅ |
| 8 | Block | 5 ms | 2 | 100 µs | 3.7 µT peak | 81 (−2:0.05:2 ppm) | −300 ppm | — | 82 | ✅ |

**Note on case 5 offset range:** The case 5 README states "−5:0.1:5 ppm" (101 points) but
the seq file contains "−2:0.02:2 ppm" (201 points). The seq file is the authoritative
simulation input created by the challenge organisers. Our 201-point output is correct.

**Note on case 8 B1:** The seq file header lists `B1rms=3.6635 µT` but the README specifies
peak power = 3.7 µT. Our simulator reads B1 amplitude directly from the RF event entries in
the seq file (157.533 Hz = 3.7 µT peak exactly), not from the header metadata.

---

## Output format

Each submission CSV has two columns: `offset_ppm` and `Mz`.

- The **first row** is the M0 normalization scan at −300 ppm (raw, un-normalised Mz)
- All remaining rows are Z-spectrum offsets in order (raw Mz, not divided by M0)
- This matches the submission format used by all other groups in the challenge spreadsheet

Example (case 1 first three rows):

```
offset_ppm,Mz
-300.0,0.9999925651
-15.0,0.9970366...
-14.9,0.9969993...
```

---

## Solver details

```
BM equation:   dM/dt = A · M + b

State vector:  [Mx_water, My_water, Mz_water,
                Mx_pool1, My_pool1, Mz_pool1,
                ...  (one 3-vector per pool, including MT)]

Steady state:  M_ss = lstsq(A, -b)           (numpy.linalg.lstsq, numerically stable)
Solution:      M(t) = expm(A·t)·(M₀-M_ss) + M_ss

expm backend:  scipy.linalg.expm
               Algorithm: Al-Mohy & Higham (2009) scaling+squaring
               Padé order: m=13 (verified for all our matrix sizes: 6×6 and 15×15)
               Norm of A·dt exceeds all lower-order Padé thresholds for our BM matrices
```

---

## Key design decisions

**Why not use BMCTool directly?**
BMCTool uses pypulseq's `seq.get_block()` which has O(n²) complexity per call and times
out for cases 1–3 (1523 blocks each). Our solver parses only the `[DEFINITIONS]` section
of the seq file for timing/offset parameters and solves the BM equations directly —
bypassing block-by-block iteration entirely.

**MT pool — 3-component vs z-only:**
The challenge FAQ explicitly states that z-only and 3-component MT implementations are
"NOT INTERCHANGEABLE". Our `add_pool()` function gives the MT pool the same full
[Mx, My, Mz] state vector as any CEST pool. The MT pool's short T2 = 40 µs
(R2 = 25,000 rad/s) causes near-instantaneous transverse dephasing, naturally
producing the Lorentzian absorption profile through the BM equations with no
separate lineshape formula needed.

**Pulse train optimisation (cases 6, 7):**
Naïve integration: 36 pulses × 301 offsets × 200 steps = 2,167,200 `expm` calls.
Our approach: precompute one full-pulse propagator matrix P(offset) per offset
(200 `expm` calls each), then apply it 36 times as a simple matrix-vector product.
Total `expm` calls: 301 × 200 = 60,200 — a 36× reduction, cutting case 6 from
~70 minutes to ~1 minute.

**ppm → rad/s conversion:**
```
Δω [rad/s] = 2π × offset_ppm × 1e-6 × γ_Hz/T × B0
```
The `1e-6` factor (ppm = parts per million) is critical. Omitting it produces
offsets of ~10⁹ rad/s (wrong by a factor of 10⁶), causing zero apparent saturation
at every offset. This bug was identified and corrected during development.

**M0 normalization convention:**
Cases 1–3 seq files use M0_offset = −1560 ppm internally. All other groups use
−300 ppm. Our code forces the M0 scan to −300 ppm for output consistency,
while still simulating the Z-spectrum using the seq file's defined offsets.
The physics difference between −300 and −1560 ppm is < 1×10⁻⁵ for these pool models.

---

## Verified output values

From actual simulation runs (raw Mz):

| Case | M0 at −300 ppm | Mz min | Mz max | Total rows |
|---|---|---|---|---|
| 1 | 0.99999257 | 0.0018 | 0.9970 | 302 |
| 2 | 0.99999341 | −0.1382 | 0.9981 | 302 |
| 3 | 0.98772247 | 0.0072 | 0.5539 | 302 |
| 4 | 0.99990364 | 0.2154 | 0.9031 | 82 |
| 5 | 0.99999991 | 0.2409 | 0.9983 | 202 |
| 6 | 0.99999759 | −0.3571 | 0.9990 | 302 |
| 7 | 0.98777427 | 0.0108 | 0.5586 | 302 |
| 8 | 0.99982215 | −0.7097 | 0.8776 | 82 |
