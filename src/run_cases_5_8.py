"""
BMsim Challenge Cases 5-8 – Optimized Runner
Precomputes per-offset pulse propagators so pulse trains run in O(n_offsets) 
instead of O(n_offsets * n_pulses).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import time
import warnings
import sys
from pathlib import Path
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).parent))
from bm_solver import (load_pool_model, parse_seq_definitions,
                        build_bm_matrix, solve_bm_delay,
                        build_m0, apply_spoiler)
from seq_helpers import parse_seq_v14, export_submission_csv_v2, plot_cases_5_8

warnings.filterwarnings("ignore")


def precompute_pulse_propagator(model, shape_samples, amplitude_hz,
                                 offset_ppm, duration_s, max_samples=50):
    """
    Compute the net matrix-exponential propagator for a single shaped pulse
    at a given offset_ppm.
    Returns: (P, c) where m_out = P @ m_in + c
    """
    N_orig = len(shape_samples)
    if N_orig > max_samples:
        factor = N_orig // max_samples
        trim   = factor * max_samples
        shape  = shape_samples[:trim].reshape(max_samples, factor).mean(axis=1)
    else:
        shape = shape_samples
        max_samples = N_orig

    N  = len(shape)
    dt = duration_s / N

    # Start with identity propagator
    n_state = len(build_m0(model))
    P_total = np.eye(n_state)
    c_total = np.zeros(n_state)

    for s in shape:
        omega1 = 2.0 * np.pi * amplitude_hz * s
        A, b   = build_bm_matrix(model, omega1, 0.0, offset_ppm)
        M_ss   = np.linalg.lstsq(A, -b, rcond=None)[0]
        eAdt   = expm(A * dt)
        # m_new = eAdt @ (m - M_ss) + M_ss = eAdt @ m + (I - eAdt) @ M_ss
        # Compose: P_new = eAdt @ P_old
        #          c_new = eAdt @ c_old + (I - eAdt) @ M_ss
        c_total = eAdt @ c_total + (np.eye(n_state) - eAdt) @ M_ss
        P_total = eAdt @ P_total

    return P_total, c_total


def precompute_delay_propagator(model, duration_s):
    """Propagator for a free-delay block (no RF)."""
    n_state = len(build_m0(model))
    A, b    = build_bm_matrix(model, 0.0, 0.0, 0.0)
    M_ss    = np.linalg.lstsq(A, -b, rcond=None)[0]
    eAdt    = expm(A * duration_s)
    c       = (np.eye(n_state) - eAdt) @ M_ss
    return eAdt, c


def simulate_case_optimized(model, seq_data, max_pulse_samples=200, verbose=False):
    """
    Optimized simulator for cases with pulse trains.
    Precomputes the pulse propagator once per offset, then applies it N_pulses times.
    """
    defs      = seq_data['definitions']
    blocks    = seq_data['blocks']
    rf_events = seq_data['rf_events']
    shapes    = seq_data['shapes']

    offsets_all = np.array(defs['offsets_ppm'])
    m0_offset   = float(defs.get('M0_offset', -300.0))
    n_cest = len(model.cest_pools)
    has_mt = model.mt_pool is not None
    m_init = build_m0(model)

    gamma_hz_per_T = model.scanner.gamma / (2.0 * np.pi) * 1e6
    larmor_hz      = gamma_hz_per_T * model.scanner.b0

    # ── Group blocks by readout (same as before) ──
    segments = []
    cur_seg  = []
    for blk in blocks:
        cur_seg.append(blk)
        if blk['adc_id'] != 0:
            segments.append(cur_seg)
            cur_seg = []

    n_seg   = len(segments)
    mz_out  = np.zeros(n_seg)

    if verbose:
        print(f"  Total readouts: {n_seg}")

    # ── Precompute recovery delay propagator (same for all offsets) ──
    # Find the recovery duration from first segment
    trec     = segments[0][0]['dur_s'] if segments else 3.5
    P_rec, c_rec = precompute_delay_propagator(model, trec)

    # ── Precompute spoiler+post-delay propagator ──
    spoi_dur = next((blk['dur_s'] for seg in segments[:1]
                     for blk in seg if blk['gz_id'] != 0), 0.0065)
    P_spoi, c_spoi = precompute_delay_propagator(model, spoi_dur)

    # ── Precompute per-offset pulse propagators ──
    if verbose:
        print(f"  Precomputing pulse propagators for {n_seg} offsets...")

    t_pre = time.time()
    pulse_props = {}  # seg_idx → (list of (P_pulse, c_pulse) per RF in segment)

    for seg_idx, seg in enumerate(segments):
        props = []
        for blk in seg:
            if blk['rf_id'] != 0:
                rf     = rf_events[blk['rf_id']]
                amp_hz = rf['amplitude_hz']
                freq   = rf['freq_hz']
                dur    = blk['dur_s']
                mag_id = rf['mag_shape_id']
                offset_ppm = freq / larmor_hz * 1e6

                if mag_id in shapes:
                    P, c = precompute_pulse_propagator(
                        model, shapes[mag_id], amp_hz, offset_ppm,
                        dur, max_samples=max_pulse_samples)
                else:
                    # Block pulse
                    omega1 = 2.0 * np.pi * amp_hz
                    A, b   = build_bm_matrix(model, omega1, 0.0, offset_ppm)
                    M_ss   = np.linalg.lstsq(A, -b, rcond=None)[0]
                    n_s    = len(m_init)
                    eAdt   = expm(A * dur)
                    c      = (np.eye(n_s) - eAdt) @ M_ss
                    P      = eAdt
                props.append(('rf', P, c, blk))
            elif blk['gz_id'] != 0:
                props.append(('spoi', None, None, blk))
            elif blk['adc_id'] != 0:
                props.append(('adc', None, None, blk))
            elif blk['dur_s'] > 1e-6:
                # Inter-pulse delay (not recovery, not spoiler)
                P_d, c_d = precompute_delay_propagator(model, blk['dur_s'])
                props.append(('delay', P_d, c_d, blk))
        pulse_props[seg_idx] = props

    if verbose:
        print(f"  Precompute done in {time.time()-t_pre:.1f}s")
        print(f"  Simulating {n_seg} readouts...")

    # ── Simulate each segment using precomputed propagators ──
    for seg_idx, seg in enumerate(segments):
        m = m_init.copy()

        for event_type, P, c, blk in pulse_props[seg_idx]:
            if event_type == 'rf':
                m = P @ m + c
            elif event_type == 'spoi':
                m = apply_spoiler(m, n_cest, has_mt)
                if blk['dur_s'] > 1e-6:
                    m = P_spoi @ m + c_spoi
            elif event_type == 'adc':
                mz_out[seg_idx] = float(m[2])
            elif event_type == 'delay':
                m = P @ m + c

        if verbose and seg_idx % max(1, n_seg // 8) == 0:
            print(f"  readout {seg_idx+1}/{n_seg}")

    # ── Extract and return raw Mz values ──
    # Return ALL offsets including M0 scan row — raw (un-normalized) Mz
    # M0 offset row comes first, then Z-spectrum offsets in original order
    m0_idx  = np.where(offsets_all == m0_offset)[0]
    mz_m0   = mz_out[m0_idx[0]] if len(m0_idx) > 0 else 1.0

    z_mask  = offsets_all != m0_offset
    z_offs  = offsets_all[z_mask]
    mz_z    = mz_out[z_mask]          # raw Mz, NOT divided by m0

    all_offsets = np.concatenate([[m0_offset], z_offs])
    all_mz      = np.concatenate([[mz_m0],     mz_z])

    return all_offsets, all_mz, float(mz_m0)


def run_case_optimized(case_num, repo_dir, verbose=True, max_pulse_samples=200):
    case_dir   = repo_dir / f"case_{case_num}"
    yaml_files = list(case_dir.glob("*.yaml"))
    seq_files  = list(case_dir.glob("*v140*.seq")) or list(case_dir.glob("*.seq"))

    if not yaml_files or not seq_files:
        raise FileNotFoundError(f"Missing files in {case_dir}")

    yaml_path = yaml_files[0]
    seq_path  = seq_files[0]

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Case {case_num}  |  {yaml_path.name}")
        print(f"  Sequence: {seq_path.name}")
        print(f"{'='*55}")

    model    = load_pool_model(yaml_path)
    seq_data = parse_seq_v14(seq_path)
    defs     = seq_data['definitions']

    offsets_all = np.array(defs['offsets_ppm'])
    m0_offset   = float(defs.get('M0_offset', -300.0))
    offsets_z   = offsets_all[offsets_all != m0_offset]

    if verbose:
        print(f"  B0: {model.scanner.b0}T  |  "
              f"n_pulses: {int(defs.get('n_pulses',1))}  |  "
              f"Offsets: {len(offsets_z)}  |  "
              f"Pools: water+{len(model.cest_pools)}CEST"
              f"{'+MT' if model.mt_pool else ''}")

    t0 = time.time()
    all_offsets, all_mz, mz_m0 = simulate_case_optimized(
        model, seq_data, max_pulse_samples=max_pulse_samples, verbose=verbose)
    elapsed = time.time() - t0

    mz_z   = all_mz[1:]      # Z-spectrum raw Mz (excluding M0 row)
    offs_z = all_offsets[1:]  # Z-spectrum offsets

    if verbose:
        print(f"\n  ✓ Done in {elapsed:.2f}s  "
              f"M0={mz_m0:.8f}  Mz=[{mz_z.min():.4f}, {mz_z.max():.4f}]")

    return {'case': case_num, 'offsets': all_offsets, 'mz': all_mz,
            'offsets_z': offs_z, 'mz_z': mz_z, 'mz_m0': mz_m0,
            'model': model, 'defs': defs, 'elapsed': elapsed,
            'yaml_path': yaml_path, 'seq_path': seq_path}


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent.resolve()
    REPO_DIR   = SCRIPT_DIR / "BMsim_challenge"
    OUTPUT_DIR = SCRIPT_DIR / "output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REPO_DIR.exists():
        REPO_DIR = Path("/home/claude/BMsim_challenge")

    print("\n" + "█"*55)
    print("  BMsim Challenge – Cases 5–8  (Optimized)")
    print("  Solver: Precomputed Propagators + Matrix Exponential")
    print("█"*55)

    results  = []
    total_t0 = time.time()

    for case_num in [5, 6, 7, 8]:
        try:
            r = run_case_optimized(case_num, REPO_DIR, verbose=True, max_pulse_samples=200)
            results.append(r)
        except Exception as e:
            print(f"  ✗ Case {case_num} failed: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*55}")
    print(f"  Total: {time.time()-total_t0:.1f}s")
    print(f"{'='*55}")

    if results:
        print("\n  Exporting CSVs...")
        export_submission_csv_v2(results, OUTPUT_DIR / "submissions")
        print("\n  Generating figure...")
        fig = plot_cases_5_8(results, OUTPUT_DIR / "zspectra_cases_5_8.png")
        plt.close(fig)
        print("\n─"*55)
        print(f"  {'Case':<6} {'Offsets':<9} {'Time(s)':<10} {'Z_min':<10} {'Z_max'}")
        print("─"*55)
        for r in results:
            print(f"  {r['case']:<6} {len(r['offsets']):<9} {r['elapsed']:<10.1f} "
                  f"{r['mz_z'].min():<10.4f} {r['mz_z'].max():.4f}")
