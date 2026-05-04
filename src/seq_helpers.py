"""
BMsim Challenge – Cases 5–8 Runner
====================================
Extends the pipeline to handle:
  Case 5: single Gaussian pulse, 2-pool
  Case 6: 36 Gaussian pulses (pulse train), 2-pool
  Case 7: 36 Gaussian pulses (pulse train), 5-pool + MT
  Case 8: 2 block pulses with 100µs interpulse delay, 5-pool + MT

Key new capability: shaped pulse integration via step-wise matrix exponential.
Each pulse shape is sampled into N time steps; we apply expm(A·dt) for each step,
re-building A at each step to account for the changing omega1(t).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import time
import warnings
from pathlib import Path
from scipy.linalg import expm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from bm_solver import (
    load_pool_model, parse_seq_definitions,
    build_bm_matrix, solve_bm_delay,
    build_m0, apply_spoiler,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────
#  Seq v1.4 parser — reads RF events, shapes, and block structure
# ─────────────────────────────────────────────────────────────────

def parse_seq_v14(seq_path: str | Path) -> dict:
    """
    Parse a Pulseq v1.4 seq file fully.

    Returns dict with:
      definitions  : key→value from [DEFINITIONS]
      blocks       : list of (block_id, duration_raster, rf_id, gz_id, adc_id)
      rf_events    : dict  rf_id → {amplitude_hz, mag_shape_id, freq_hz, phase_rad}
      shapes       : dict  shape_id → np.array of normalized samples (0..1)
      raster       : BlockDurationRaster in seconds
    """
    lines = [l.rstrip('\r\n') for l in open(seq_path)]

    # ── Find section start lines ──
    sec = {}
    for i, l in enumerate(lines):
        ls = l.strip()
        if ls.startswith('[') and ls.endswith(']'):
            sec[ls] = i

    # ── Parse [DEFINITIONS] ──
    definitions = {}
    if '[DEFINITIONS]' in sec:
        for l in lines[sec['[DEFINITIONS]'] + 1:]:
            ls = l.strip()
            if ls.startswith('['):
                break
            if ls:
                parts = ls.split(None, 1)
                if len(parts) == 2:
                    key, val = parts
                    vals = val.strip().split()
                    if len(vals) > 1:
                        try:    definitions[key] = [float(v) for v in vals]
                        except: definitions[key] = vals
                    else:
                        try:    definitions[key] = float(vals[0])
                        except: definitions[key] = vals[0]

    raster = float(definitions.get('BlockDurationRaster', 1e-5))

    # ── Parse [BLOCKS] ──
    # Format: NUM DUR RF GX GY GZ ADC EXT
    blocks = []
    if '[BLOCKS]' in sec:
        for l in lines[sec['[BLOCKS]'] + 1:]:
            ls = l.strip()
            if ls.startswith('[') or ls.startswith('#'):
                break
            if ls:
                try:
                    parts = ls.split()
                    if len(parts) >= 7:
                        blocks.append({
                            'id':       int(parts[0]),
                            'dur_s':    int(parts[1]) * raster,
                            'rf_id':    int(parts[2]),
                            'gz_id':    int(parts[5]),
                            'adc_id':   int(parts[6]),
                        })
                except:
                    pass

    # ── Parse [RF] ──
    # Format: id amplitude_Hz mag_shape_id phase_shape_id time_shape_id delay freq_Hz phase_rad
    rf_events = {}
    if '[RF]' in sec:
        for l in lines[sec['[RF]'] + 1:]:
            ls = l.strip()
            if ls.startswith('[') or ls.startswith('#'):
                break
            if ls:
                try:
                    parts = ls.split()
                    if len(parts) >= 8:
                        rf_id = int(parts[0])
                        rf_events[rf_id] = {
                            'amplitude_hz':  float(parts[1]),
                            'mag_shape_id':  int(parts[2]),
                            'phase_shape_id':int(parts[3]),
                            'freq_hz':       float(parts[6]),   # freq offset [Hz]
                            'phase_rad':     float(parts[7]),
                        }
                except:
                    pass

    # ── Parse [SHAPES] ──
    shapes = {}
    if '[SHAPES]' in sec:
        cur_id = None
        cur_n  = None
        cur_samples = []
        for l in lines[sec['[SHAPES]'] + 1:]:
            ls = l.strip()
            if ls.startswith('['):
                break
            if ls.startswith('shape_id'):
                if cur_id is not None and cur_samples:
                    shapes[cur_id] = np.array(cur_samples[:cur_n])
                cur_id = int(ls.split()[1])
                cur_samples = []
                cur_n = None
            elif ls.startswith('num_samples'):
                cur_n = int(ls.split()[1])
            elif ls:
                try:
                    cur_samples.append(float(ls))
                except:
                    pass
        if cur_id is not None and cur_samples:
            shapes[cur_id] = np.array(cur_samples[:cur_n])

    return {
        'definitions': definitions,
        'blocks':      blocks,
        'rf_events':   rf_events,
        'shapes':      shapes,
        'raster':      raster,
    }


# ─────────────────────────────────────────────────────────────────
#  Shaped pulse integrator
# ─────────────────────────────────────────────────────────────────

def simulate_shaped_pulse(model, shape_samples: np.ndarray, amplitude_hz: float,
                           freq_offset_hz: float, duration_s: float,
                           m_init: np.ndarray, max_samples: int = 100) -> np.ndarray:
    """
    Integrate the BM equations through a shaped RF pulse using step-wise expm.

    shape_samples : normalized envelope (0..1), length N
    amplitude_hz  : peak B1 amplitude in Hz  (= gamma_hz * B1peak_uT)
    freq_offset_hz: saturation frequency offset [Hz]
    duration_s    : total pulse duration [s]
    max_samples   : downsample shape to this many steps for speed
    """
    gamma = model.scanner.gamma          # rad/(s·µT)
    gamma_hz_uT = gamma / (2.0 * np.pi) # Hz/µT

    # Downsample shape for speed (while preserving B1rms)
    N_orig = len(shape_samples)
    if N_orig > max_samples:
        # Bin-average to preserve RMS
        factor = N_orig // max_samples
        trim = factor * max_samples
        shape_ds = shape_samples[:trim].reshape(max_samples, factor).mean(axis=1)
    else:
        shape_ds = shape_samples
        max_samples = N_orig

    N   = len(shape_ds)
    dt  = duration_s / N

    # Convert freq offset to ppm
    gamma_hz_per_T = gamma / (2.0 * np.pi) * 1e6
    larmor_hz = gamma_hz_per_T * model.scanner.b0
    offset_ppm = freq_offset_hz / larmor_hz * 1e6

    m = m_init.copy()
    for s in shape_ds:
        # Instantaneous omega1 [rad/s]: amplitude_hz is in Hz → *2π → rad/s, *shape
        omega1_step = 2.0 * np.pi * amplitude_hz * s
        A, b = build_bm_matrix(model, omega1_step, 0.0, offset_ppm)
        M_ss = np.linalg.lstsq(A, -b, rcond=None)[0]
        eAdt = expm(A * dt)
        m    = eAdt @ (m - M_ss) + M_ss

    return m


# ─────────────────────────────────────────────────────────────────
#  Full sequence simulator (reads the .seq file directly)
# ─────────────────────────────────────────────────────────────────

def simulate_zspectrum_from_seq(model, seq_data: dict,
                                 max_pulse_samples: int = 100,
                                 verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the full Z-spectrum by walking through the seq file block-by-block.

    Handles: recovery delays, shaped RF pulses, pulse trains,
             spoiler gradients, and ADC readouts.
    """
    defs        = seq_data['definitions']
    blocks      = seq_data['blocks']
    rf_events   = seq_data['rf_events']
    shapes      = seq_data['shapes']

    offsets_all = np.array(defs['offsets_ppm'])
    m0_offset   = float(defs.get('M0_offset', -300.0))
    offsets_z   = offsets_all[offsets_all != m0_offset]

    n_cest = len(model.cest_pools)
    has_mt = model.mt_pool is not None
    m_init = build_m0(model)

    # ── Walk blocks once and group them by readout ──
    # Each "segment" ends with an ADC block. We collect all blocks between ADCs.
    segments = []   # list of lists of blocks
    cur_seg  = []
    for blk in blocks:
        cur_seg.append(blk)
        if blk['adc_id'] != 0:
            segments.append(cur_seg)
            cur_seg = []
    if cur_seg:
        segments.append(cur_seg)

    n_seg = len(segments)
    if verbose:
        print(f"  Total readouts (segments): {n_seg}")

    mz_out = np.zeros(n_seg)

    for seg_idx, seg in enumerate(segments):
        m = m_init.copy()

        for blk in seg:
            dur = blk['dur_s']

            if blk['adc_id'] != 0:
                # ADC: record Mz_water
                mz_out[seg_idx] = float(m[2])

            elif blk['rf_id'] != 0:
                # RF pulse (shaped or block)
                rf = rf_events[blk['rf_id']]
                mag_id = rf['mag_shape_id']
                amp_hz = rf['amplitude_hz']
                freq_hz = rf['freq_hz']

                if mag_id in shapes:
                    # Shaped pulse
                    m = simulate_shaped_pulse(
                        model, shapes[mag_id], amp_hz, freq_hz, dur, m,
                        max_samples=max_pulse_samples)
                else:
                    # Block pulse (flat shape = 1.0)
                    gamma_hz_per_T = model.scanner.gamma / (2.0 * np.pi) * 1e6
                    larmor_hz = gamma_hz_per_T * model.scanner.b0
                    offset_ppm = freq_hz / larmor_hz * 1e6
                    omega1 = 2.0 * np.pi * amp_hz
                    from bm_solver import solve_bm_cw
                    m = solve_bm_cw(model, omega1, 0.0, offset_ppm, dur, m)

            elif blk['gz_id'] != 0:
                # Spoiler gradient
                m = apply_spoiler(m, n_cest, has_mt)
                if dur > 1e-6:
                    m = solve_bm_delay(model, dur, m)

            elif dur > 1e-6:
                # Pure delay / recovery
                m = solve_bm_delay(model, dur, m)

        if verbose and seg_idx % max(1, n_seg // 10) == 0:
            print(f"  readout {seg_idx+1}/{n_seg}")

    # ── Extract M0 and Z-spectrum ──
    all_offsets = offsets_all

    # M0 is the readout corresponding to m0_offset
    m0_idx  = np.where(all_offsets == m0_offset)[0]
    m0_val  = mz_out[m0_idx[0]] if len(m0_idx) > 0 else 1.0

    # Z-spectrum offsets (excluding M0 scan)
    z_mask  = all_offsets != m0_offset
    z_offs  = all_offsets[z_mask]
    z_vals  = mz_out[z_mask] / m0_val

    return z_offs, z_vals, float(m0_val)


# ─────────────────────────────────────────────────────────────────
#  Per-case runner for cases 5–8
# ─────────────────────────────────────────────────────────────────

def run_case_v14(case_num: int, repo_dir: Path, verbose: bool = True,
                  max_pulse_samples: int = 100) -> dict:
    """Run one case using the v1.4 seq file parser and shaped pulse integrator."""
    case_dir  = repo_dir / f"case_{case_num}"
    yaml_files = list(case_dir.glob("*.yaml"))

    # Prefer v1.4 seq files
    seq_files_140 = list(case_dir.glob("*v140*.seq"))
    seq_files_131 = list(case_dir.glob("*v131*.seq"))
    seq_files = seq_files_140 if seq_files_140 else seq_files_131
    if not seq_files:
        seq_files = list(case_dir.glob("*.seq"))

    if not yaml_files or not seq_files:
        raise FileNotFoundError(f"Missing YAML or SEQ in {case_dir}")

    yaml_path = yaml_files[0]
    seq_path  = seq_files[0]

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Case {case_num}")
        print(f"  Pool model : {yaml_path.name}")
        print(f"  Sequence   : {seq_path.name}")
        print(f"{'='*55}")

    model    = load_pool_model(yaml_path)
    seq_data = parse_seq_v14(seq_path)
    defs     = seq_data['definitions']

    offsets_all = np.array(defs['offsets_ppm'])
    m0_offset   = float(defs.get('M0_offset', -300.0))
    offsets_z   = offsets_all[offsets_all != m0_offset]
    n_pulses    = int(defs.get('n_pulses', 1))
    t_sat       = float(defs.get('Tsat', defs.get('tp', 0.05)))
    b1rms       = float(defs.get('B1rms', defs.get('B1pa', 2.0)))

    if verbose:
        print(f"  B0         : {model.scanner.b0} T")
        print(f"  B1rms      : {b1rms:.4f} µT")
        print(f"  T_sat      : {t_sat} s")
        print(f"  n_pulses   : {n_pulses}")
        print(f"  Offsets    : {len(offsets_z)} points  "
              f"({offsets_z.min():.1f} to {offsets_z.max():.1f} ppm)")
        print(f"  Pools      : water + {len(model.cest_pools)} CEST"
              f"{' + MT' if model.mt_pool else ''}")
        print(f"  Pulse samples/step: {max_pulse_samples}")
        print()

    t0 = time.time()
    offsets, z_spectrum, m0_val = simulate_zspectrum_from_seq(
        model, seq_data, max_pulse_samples=max_pulse_samples, verbose=verbose)
    elapsed = time.time() - t0

    if verbose:
        print(f"\n  ✓ Done in {elapsed:.2f}s")
        print(f"  M0 value : {m0_val:.6f}")
        print(f"  Z range  : [{z_spectrum.min():.4f}, {z_spectrum.max():.4f}]")

    return {
        'case':       case_num,
        'offsets':    offsets,
        'z_spectrum': z_spectrum,
        'm0_val':     m0_val,
        'model':      model,
        'defs':       defs,
        'elapsed':    elapsed,
        'yaml_path':  yaml_path,
        'seq_path':   seq_path,
    }


# ─────────────────────────────────────────────────────────────────
#  Plot cases 5–8
# ─────────────────────────────────────────────────────────────────

def plot_cases_5_8(results: list[dict], output_path: Path):
    n    = len(results)
    cols = 2
    rows = (n + 1) // cols

    fig = plt.figure(figsize=(14, 5 * rows), facecolor="white")
    fig.suptitle("BMsim Challenge – Simulated Z-Spectra (Cases 5–8)\n"
                 "Shaped Pulse / Pulse Train  |  Step-wise Matrix Exponential",
                 fontsize=13, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.45, wspace=0.35)

    colors = ["#0891B2", "#7C3AED", "#059669", "#D97706"]

    case_labels = {
        5: "Case 5 – 2-pool, 1× Gaussian 50ms, B1rms=2µT\n(creatine phantom)",
        6: "Case 6 – 2-pool, 36× Gaussian 50ms, B1rms=2µT\n(APTw pulse train, creatine phantom)",
        7: "Case 7 – 5-pool, 36× Gaussian 50ms, B1rms=2µT\n(APTw pulse train, WM model)",
        8: "Case 8 – 5-pool, 2× Block 5ms, B1=3.7µT\n(modified WASABI, WM model)",
    }

    for idx, r in enumerate(results):
        ax       = fig.add_subplot(gs[idx // cols, idx % cols])
        c        = colors[idx]
        case_num = r["case"]

        ax.plot(r["offsets"], r["z_spectrum"], color=c, lw=2)
        ax.set_xlabel("Saturation Offset (ppm)", fontsize=10)
        ax.set_ylabel("Z = Mz / M0", fontsize=10)
        ax.set_title(case_labels.get(case_num, f"Case {case_num}"),
                     fontsize=10, fontweight="bold")
        ax.invert_xaxis()
        ax.set_ylim(-0.05, 1.12)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")

        ax.text(0.03, 0.04, f"⏱ {r['elapsed']:.1f}s",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8, color="gray")

        # Pool labels with leader lines
        sorted_pools = sorted(r["model"].cest_pools, key=lambda p: p.dw, reverse=True)
        n_pools = len(sorted_pools)
        xlo, xhi = ax.get_xlim()
        x_span   = xlo - xhi
        margin   = x_span * 0.1
        if n_pools == 1:
            anchors = [xlo - x_span * 0.5]
        else:
            anchors = [xlo - margin - i * (x_span - 2 * margin) / (n_pools - 1)
                       for i in range(n_pools)]

        for cp, anchor_x in zip(sorted_pools, anchors):
            ax.axvline(cp.dw, color=c, lw=0.8, ls=":", alpha=0.5)
            ax.annotate(
                f"{cp.name.split('_')[0]}  {cp.dw} ppm",
                xy=(cp.dw, -0.04), xytext=(anchor_x, -0.14),
                fontsize=6.5, color=c, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color=c, alpha=0.5, lw=0.8),
                bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.85),
            )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n  Figure saved: {output_path}")
    return fig


# ─────────────────────────────────────────────────────────────────
#  Export submission CSVs
# ─────────────────────────────────────────────────────────────────

def export_submission_csv_v2(results: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        df   = pd.DataFrame({'offset_ppm': r['offsets'], 'Mz_norm': r['z_spectrum']})
        path = output_dir / f"case_{r['case']}_submission.csv"
        df.to_csv(path, index=False, float_format="%.8f")
        print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent.resolve()
    REPO_DIR   = SCRIPT_DIR / "BMsim_challenge"
    OUTPUT_DIR = SCRIPT_DIR / "output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REPO_DIR.exists():
        print(f"✗ Repo not found at {REPO_DIR}")
        sys.exit(1)

    print("\n" + "█"*55)
    print("  BMsim Challenge – Cases 5–8 Pipeline")
    print("  Solver: Step-wise Matrix Exponential (Shaped Pulses)")
    print("█"*55)
    print(f"\n  Repo dir   : {REPO_DIR}")
    print(f"  Output dir : {OUTPUT_DIR}")

    results = []
    total_t0 = time.time()

    for case_num in [5, 6, 7, 8]:
        try:
            r = run_case_v14(case_num, REPO_DIR, verbose=True, max_pulse_samples=100)
            results.append(r)
        except Exception as e:
            print(f"  ✗ Case {case_num} failed: {e}")
            import traceback; traceback.print_exc()

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*55}")
    print(f"  All cases done in {total_elapsed:.1f}s total")
    print(f"{'='*55}")

    if not results:
        print("\n  ✗ No cases succeeded.")
        sys.exit(1)

    print("\n  Exporting submission files...")
    export_submission_csv_v2(results, OUTPUT_DIR / "submissions")

    print("\n  Generating figures...")
    fig = plot_cases_5_8(results, OUTPUT_DIR / "zspectra_cases_5_8.png")
    plt.close(fig)

    print("\n" + "─"*55)
    print(f"  {'Case':<6} {'Offsets':<9} {'Time(s)':<10} {'Z_min':<10} {'Z_max'}")
    print("─"*55)
    for r in results:
        print(f"  {r['case']:<6} {len(r['offsets']):<9} {r['elapsed']:<10.1f} "
              f"{r['z_spectrum'].min():<10.4f} {r['z_spectrum'].max():.4f}")
    print("─"*55)
    print(f"\n  Output directory: {OUTPUT_DIR}/")
