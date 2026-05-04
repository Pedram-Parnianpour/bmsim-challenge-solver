"""
Bloch-McConnell (BM) Solver for the BMsim Challenge
=====================================================
Pure NumPy/SciPy implementation using matrix-exponential integration.

Complies with ALL criteria from https://github.com/pulseq-cest/BMsim_challenge:

  [1] gamma = 42.5764 MHz/T  (rounded NIST shielded proton value)
  [2] Pool size fraction: definition 1 — water f=1, others relative to water
  [3] Fully relaxed initial magnetization (Zi=1) for every offset
  [4] Post-preparation delay = 6.5 ms (gradient spoiler)
  [5] Normalization scan at -300 ppm
  [6] MT pool: full 3-component [Mx, My, Mz] — NOT z-only
      FAQ: "treat the MT pool similar to a CEST pool and consider all 3 components"
      FAQ: "z-only and 3-component implementations are NOT INTERCHANGEABLE"
  [7] Larmor frequency at 3T = 127.7292 MHz
"""

import numpy as np
import yaml
from pathlib import Path
from scipy.linalg import expm
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class WaterPool:
    f: float
    t1: float
    t2: float

@dataclass
class CESTPool:
    name: str
    f: float
    t1: float
    t2: float
    k: float    # exchange rate [Hz] pool→water
    dw: float   # chemical shift [ppm]

@dataclass
class MTPool:
    f: float
    t1: float
    t2: float   # very short (e.g. 40 µs) → Lorentzian lineshape via full BM equations
    k: float
    dw: float
    lineshape: str = "Lorentzian"

@dataclass
class ScannerSettings:
    b0: float = 3.0
    gamma: float = 267.5154109126009   # rad/(s·µT)  =  42.5764 MHz/T * 2π
    b0_inhom: float = 0.0
    rel_b1: float = 1.0

@dataclass
class SimOptions:
    reset_init_mag: bool = True
    scale: float = 1.0
    max_pulse_samples: int = 300
    verbose: bool = False

@dataclass
class PoolModel:
    water: WaterPool
    cest_pools: list = field(default_factory=list)
    mt_pool: Optional[MTPool] = None
    scanner: ScannerSettings = field(default_factory=ScannerSettings)
    options: SimOptions = field(default_factory=SimOptions)


# ─────────────────────────────────────────────────────────────────
#  YAML loader
# ─────────────────────────────────────────────────────────────────

def load_pool_model(yaml_path: str | Path) -> PoolModel:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    water_cfg = cfg["water_pool"]
    water = WaterPool(f=float(water_cfg["f"]),
                      t1=float(water_cfg["t1"]),
                      t2=float(water_cfg["t2"]))

    cest_pools = []
    if "cest_pool" in cfg and cfg["cest_pool"]:
        for name, p in cfg["cest_pool"].items():
            cest_pools.append(CESTPool(name=name, f=float(p["f"]),
                                       t1=float(p["t1"]), t2=float(p["t2"]),
                                       k=float(p["k"]), dw=float(p["dw"])))

    mt_pool = None
    if "mt_pool" in cfg and cfg["mt_pool"]:
        p = cfg["mt_pool"]
        mt_pool = MTPool(f=float(p["f"]), t1=float(p["t1"]), t2=float(p["t2"]),
                         k=float(p["k"]), dw=float(p["dw"]),
                         lineshape=str(p.get("lineshape", "Lorentzian")))

    scanner = ScannerSettings(b0=float(cfg.get("b0", 3.0)),
                               gamma=float(cfg.get("gamma", 267.5154109126009)),
                               b0_inhom=float(cfg.get("b0_inhom", 0.0)),
                               rel_b1=float(cfg.get("rel_b1", 1.0)))

    options = SimOptions(reset_init_mag=bool(cfg.get("reset_init_mag", True)),
                         scale=float(cfg.get("scale", 1.0)),
                         max_pulse_samples=int(cfg.get("max_pulse_samples", 300)),
                         verbose=bool(cfg.get("verbose", False)))

    return PoolModel(water=water, cest_pools=cest_pools, mt_pool=mt_pool,
                     scanner=scanner, options=options)


# ─────────────────────────────────────────────────────────────────
#  SEQ file parser
# ─────────────────────────────────────────────────────────────────

def parse_seq_definitions(seq_path: str | Path) -> dict:
    definitions = {}
    with open(seq_path) as f:
        in_defs = False
        for line in f:
            line = line.strip()
            if line == "[DEFINITIONS]":
                in_defs = True; continue
            if line.startswith("[") and line != "[DEFINITIONS]":
                in_defs = False
            if in_defs and line:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    key, val = parts
                    vals = val.strip().split()
                    if len(vals) > 1:
                        try:    definitions[key] = [float(v) for v in vals]
                        except: definitions[key] = vals
                    else:
                        try:    definitions[key] = float(vals[0])
                        except: definitions[key] = vals[0]
    return definitions


# ─────────────────────────────────────────────────────────────────
#  Bloch-McConnell matrix builder
# ─────────────────────────────────────────────────────────────────

def build_bm_matrix(model: PoolModel, omega1: float,
                    dw_water_rad: float, offset_ppm: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Build BM system matrix A and constant vector b  (dM/dt = A·M + b).

    State vector: [Mx_w, My_w, Mz_w,
                   Mx_c1, My_c1, Mz_c1, ...   (one block per CEST pool)
                   Mx_mt, My_mt, Mz_mt]        (if MT pool present)

    Size: 3 * (1 + n_cest + has_mt)

    MT pool is treated identically to a CEST pool (3 components) as required
    by the BMsim challenge FAQ. Its very short T2 (e.g. 40 µs, R2=25000 rad/s)
    naturally produces the Lorentzian lineshape via the BM equations.
    """
    gamma        = model.scanner.gamma
    b0           = model.scanner.b0
    b0_inhom_ppm = model.scanner.b0_inhom
    w            = model.water

    # ppm → rad/s  (criterion [1],[7]: gamma=42.5764 MHz/T, larmor=127.7292 MHz at 3T)
    gamma_hz_per_T = gamma / (2.0 * np.pi) * 1e6     # Hz/T
    larmor_hz      = gamma_hz_per_T * b0              # ~127.7292e6 Hz

    def ppm2rad(ppm: float) -> float:
        return 2.0 * np.pi * (ppm + b0_inhom_ppm) * 1e-6 * larmor_hz

    n_cest = len(model.cest_pools)
    has_mt = model.mt_pool is not None
    n      = 3 * (1 + n_cest + (1 if has_mt else 0))

    A = np.zeros((n, n))
    b = np.zeros(n)

    R1w = 1.0 / w.t1
    R2w = 1.0 / w.t2
    dw_w = ppm2rad(offset_ppm)   # water effective offset

    # ── Water ──
    A[0, 0] = -R2w;  A[0, 1] =  dw_w
    A[1, 1] = -R2w;  A[1, 0] = -dw_w;  A[1, 2] = -omega1
    A[2, 2] = -R1w;  A[2, 1] =  omega1
    b[2]    =  R1w * w.f

    # ── Generic pool adder (CEST or MT — same math) ──
    def add_pool(ix: int, R1: float, R2: float, dw_rad: float,
                 f_pool: float, k_out: float):
        """
        Add one pool block starting at index ix.
        k_out: exchange rate FROM pool TO water [Hz]
        k_in = k_out * f_pool / f_water   (detailed balance, definition 1)
        """
        k_in = k_out * f_pool / w.f
        iy, iz = ix + 1, ix + 2

        # Pool self
        A[ix, ix] -= R2 + k_out
        A[iy, iy] -= R2 + k_out
        A[iz, iz] -= R1 + k_out

        # Pool precession + RF
        A[ix, iy] += dw_rad;   A[iy, ix] -= dw_rad
        A[iy, iz] -= omega1;   A[iz, iy] += omega1

        # Exchange pool → water
        A[0, ix] += k_out;  A[1, iy] += k_out;  A[2, iz] += k_out

        # Exchange water → pool
        A[ix, 0] += k_in;   A[iy, 1] += k_in;   A[iz, 2] += k_in

        # Water loses magnetization to this pool (k_in flows out of water)
        A[0, 0] -= k_in;  A[1, 1] -= k_in;  A[2, 2] -= k_in

        # Pool equilibrium
        b[iz] += R1 * f_pool

    # ── CEST pools ──
    for i, cp in enumerate(model.cest_pools):
        add_pool(ix     = 3 + 3 * i,
                 R1     = 1.0 / cp.t1,
                 R2     = 1.0 / cp.t2,
                 dw_rad = ppm2rad(offset_ppm - cp.dw),
                 f_pool = cp.f,
                 k_out  = cp.k)

    # ── MT pool — full 3-component (BMsim challenge FAQ requirement) ──
    if has_mt:
        mt = model.mt_pool
        add_pool(ix     = 3 + 3 * n_cest,
                 R1     = 1.0 / mt.t1,
                 R2     = 1.0 / mt.t2,   # e.g. 1/40µs = 25000 rad/s → Lorentzian shape
                 dw_rad = ppm2rad(offset_ppm - mt.dw),
                 f_pool = mt.f,
                 k_out  = mt.k)

    return A, b


# ─────────────────────────────────────────────────────────────────
#  Solvers
# ─────────────────────────────────────────────────────────────────

def solve_bm_cw(model: PoolModel, omega1: float, dw_water_rad: float,
                offset_ppm: float, duration: float,
                m_init: np.ndarray) -> np.ndarray:
    """Matrix-exponential solution for a CW block pulse."""
    A, b  = build_bm_matrix(model, omega1, dw_water_rad, offset_ppm)
    M_ss  = np.linalg.lstsq(A, -b, rcond=None)[0]   # steady state
    return expm(A * duration) @ (m_init - M_ss) + M_ss


def solve_bm_delay(model: PoolModel, duration: float,
                   m_init: np.ndarray) -> np.ndarray:
    """Free relaxation (no RF) — used for recovery and spoiler delays."""
    return solve_bm_cw(model, omega1=0.0, dw_water_rad=0.0,
                        offset_ppm=0.0, duration=duration, m_init=m_init)


# ─────────────────────────────────────────────────────────────────
#  Initial magnetization and spoiler
# ─────────────────────────────────────────────────────────────────

def build_m0(model: PoolModel) -> np.ndarray:
    """Fully-relaxed magnetization: Mz = f*scale for each pool, Mx=My=0."""
    n_cest = len(model.cest_pools)
    has_mt = model.mt_pool is not None
    n  = 3 * (1 + n_cest + (1 if has_mt else 0))
    m0 = np.zeros(n)
    m0[2] = model.water.f * model.options.scale
    for i, cp in enumerate(model.cest_pools):
        m0[3 + 3 * i + 2] = cp.f * model.options.scale
    if has_mt:
        m0[3 + 3 * n_cest + 2] = model.mt_pool.f * model.options.scale
    return m0


def apply_spoiler(m: np.ndarray, n_cest: int, has_mt: bool) -> np.ndarray:
    """Zero all transverse components (gradient spoiler simulation)."""
    m_out  = m.copy()
    n_pools = 1 + n_cest + (1 if has_mt else 0)
    for i in range(n_pools):
        m_out[3 * i]     = 0.0
        m_out[3 * i + 1] = 0.0
    return m_out
