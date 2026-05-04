# Bloch-McConnell Simulation — BMsim Challenge

Pure Python implementation of Bloch-McConnell (BM) equations for the
[BMsim Challenge](https://github.com/pulseq-cest/BMsim_challenge) —
a community validation study of CEST MRI simulation tools.

---

## What this is

The BMsim Challenge asks research groups worldwide to simulate 8 well-defined
CEST MRI scenarios and compare results. This repository contains a fully
spec-compliant Python solver that:

- Implements the full multi-pool Bloch-McConnell equations
- Uses **matrix-exponential integration** (scipy `expm`, Padé order 13)
- Covers all 8 cases: CW block pulses (cases 1–4) and shaped/pulsed sequences (cases 5–8)
- Complies with every requirement in the [challenge FAQ](https://github.com/pulseq-cest/BMsim_challenge#faq)

## Simulation tool details

| Property | Value |
|---|---|
| Language | Python 3.12 (NumPy / SciPy) |
| Number of pools | Dynamic (2-pool or 5-pool per case, read from YAML) |
| MT components | x, y, z (full 3-component treatment) |
| MT lineshape | Lorentzian (via T2 = 40 µs in full BM equations) |
| f_water = 1 | Yes (definition 1) |
| Solver | `scipy.linalg.expm` — Al-Mohy & Higham (2009) scaling & squaring, Padé order m=13 |
| Homogeneous matrix | No — steady-state shift computed separately: M(t) = expm(A·t)·(M₀ − M_ss) + M_ss |

---

## Repository structure

```
bmsim_challenge_solver/
├── src/
│   ├── bm_solver.py          # Core BM physics library
│   ├── run_simulations.py    # Runner for cases 1–4 (CW block pulses)
│   ├── run_cases_5_8.py      # Runner for cases 5–8 (shaped pulses / pulse trains)
│   └── seq_helpers.py        # Seq v1.4 parser, shaped pulse integrator, plotting
├── results/
│   ├── case_1_submission.csv
│   ├── case_2_submission.csv
│   ├── case_3_submission.csv
│   ├── case_4_submission.csv
│   ├── case_5_submission.csv
│   ├── case_6_submission.csv
│   ├── case_7_submission.csv
│   └── case_8_submission.csv
├── figures/
│   ├── zspectra_cases_1_4.png
│   └── zspectra_cases_5_8.png
└── docs/
    └── audit_report.md
```

---

## Cases covered

### Study 1 — CW block pulses

| Case | Model | Sequence | Offsets |
|---|---|---|---|
| 1 | 2-pool creatine phantom | Block 15s, 2µT (steady-state) | −15:0.1:15 ppm |
| 2 | 2-pool creatine phantom | Block 2s, 2µT (APTw) | −15:0.1:15 ppm |
| 3 | 5-pool white matter | Block 2s, 2µT (APTw) | −15:0.1:15 ppm |
| 4 | 5-pool white matter | Block 5ms, 3.7µT (WASABI) | −2:0.05:2 ppm |

### Study 2 — Shaped pulses and pulse trains

| Case | Model | Sequence | Offsets |
|---|---|---|---|
| 5 | 2-pool creatine phantom | 1× Gaussian 50ms, B1rms=2µT | −2:0.02:2 ppm |
| 6 | 2-pool creatine phantom | 36× Gaussian 50ms, B1rms=2µT (APTw train) | −15:0.1:15 ppm |
| 7 | 5-pool white matter | 36× Gaussian 50ms, B1rms=2µT (APTw train) | −15:0.1:15 ppm |
| 8 | 5-pool white matter | 2× Block 5ms, 3.7µT (modified WASABI) | −2:0.05:2 ppm |

---

## Installation

```bash
# Clone this repo
git clone https://github.com/YOUR_USERNAME/bmsim_challenge_solver.git
cd bmsim_challenge_solver

# Clone the challenge data (pool models + seq files)
git clone https://github.com/pulseq-cest/BMsim_challenge.git

# Install dependencies
pip install numpy scipy matplotlib pandas pyyaml
```

---

## Running the simulations

```bash
cd src

# Cases 1–4 (fast, ~1 second total)
python3 run_simulations.py

# Cases 5–8 (cases 6 and 7 take ~1–3 minutes each)
python3 run_cases_5_8.py
```

Output files are written to `../output/`:
- `submissions/case_N_submission.csv` — Z-spectrum ready for the Google Sheet
- `zspectra_cases_1_4.png` — figure of all 4 CW Z-spectra
- `zspectra_cases_5_8.png` — figure of all 4 shaped-pulse Z-spectra

---

## Physics

The Bloch-McConnell equations for an N-pool system are:

```
dM/dt = A·M + b
```

where `A` is the (3N × 3N) system matrix encoding relaxation, exchange,
RF coupling, and precession for each pool, and `b` is the equilibrium
recovery vector.

The solution is:

```
M(t) = expm(A·t) · (M₀ − M_ss) + M_ss
```

where `M_ss = −A⁻¹·b` is the steady-state magnetization.

For shaped pulses (cases 5–8), the pulse envelope is sampled into 200
time steps. A propagator matrix `P = expm(A·Δt)` is computed per step,
and steps are composed to form the full pulse propagator — computed once
per offset frequency and reused across all pulses in a train.

### MT pool

The MT pool is treated as a full 3-component [Mx, My, Mz] pool identical
in structure to a CEST pool. Its very short T2 (40 µs → R2 = 25,000 rad/s)
naturally produces the Lorentzian absorption profile through the BM equations,
without any explicit lineshape term. This is the recommended approach per the
[BMsim challenge FAQ](https://github.com/pulseq-cest/BMsim_challenge#faq).

---

## Compliance with BMsim challenge criteria

| Criterion | Status |
|---|---|
| γ = 42.5764 MHz/T (NIST rounded) | ✅ |
| Water f = 1 (definition 1) | ✅ |
| Fully relaxed Zi = 1 per offset | ✅ |
| Post-prep delay = 6.5 ms | ✅ |
| MT pool: full 3-component [Mx, My, Mz] | ✅ |
| Larmor = 127.7292 MHz at 3T | ✅ |
| max_pulse_samples = 200 | ✅ |
| Normalization scan at far off-resonance | ✅ |

---

## Results

All 8 Z-spectra are shown below.

**Cases 1–4** (CW block pulses):

![Z-spectra cases 1-4](figures/zspectra_cases_1_4.png)

**Cases 5–8** (shaped pulses and pulse trains):

![Z-spectra cases 5-8](figures/zspectra_cases_5_8.png)

---

## References

- Woessner et al., *MRM* 2005 — Multi-pool BM equations
- Zaiss & Bachert, *NMR Biomed* 2013 — CEST MRI overview
- Al-Mohy & Higham, *SIAM J. Matrix Anal.* 2009 — Matrix exponential algorithm
- BMsim Challenge: https://github.com/pulseq-cest/BMsim_challenge
- pulseq-CEST library: https://github.com/pulseq-cest/pulseq-cest-library
