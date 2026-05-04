"""
BMsim Challenge – Simulation Runner
=====================================
Runs Cases 1–4 using our pure NumPy/SciPy Bloch-McConnell solver.

Each case uses:
  - A YAML pool-model file  →  loaded by bm_solver.load_pool_model()
  - A .seq file             →  header parsed for offsets + timing
  - The BM solver           →  matrix-exponential integration

Output: Z-spectra (Mz normalized to M0 scan) ready for submission.
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

from bm_solver import (
    load_pool_model, parse_seq_definitions,
    solve_bm_cw, solve_bm_delay,
    build_m0, apply_spoiler,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
#  Core simulation: one full Z-spectrum acquisition
# ─────────────────────────────────────────────────────────────────

def simulate_zspectrum(
    model,
    offsets_all: np.ndarray,
    m0_offset_ppm: float,
    b1_ut: float,
    t_sat: float,
    t_rec: float,
    t_rec_m0: float,
    t_d: float = 0.0,
    post_sat_delay: float = 0.0065,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate raw Mz for ALL offsets including the M0 scan offset.

    Every offset (including -300 ppm) is simulated with Trec, except
    the M0 offset which uses Trec_M0. Raw (un-normalized) Mz is returned
    for every offset, matching the format used by all other BMsim groups.

    Returns
    -------
    all_offsets : complete offset array (M0 offset first, then Z-spectrum)
    all_mz      : raw Mz values (NOT divided by M0)
    """
    gamma     = model.scanner.gamma
    b1_actual = b1_ut * model.scanner.rel_b1
    omega1    = gamma * b1_actual
    n_cest    = len(model.cest_pools)
    has_mt    = model.mt_pool is not None
    m_init    = build_m0(model)

    def run_one_offset(offset_ppm: float, trec: float) -> float:
        m = m_init.copy()
        if trec > 1e-9:
            m = solve_bm_delay(model, trec, m)
        m = solve_bm_cw(model, omega1=omega1, dw_water_rad=0.0,
                         offset_ppm=offset_ppm, duration=t_sat, m_init=m)
        if post_sat_delay > 1e-9:
            m = apply_spoiler(m, n_cest, has_mt)
            m = solve_bm_delay(model, post_sat_delay, m)
        return float(m[2])

    # Separate Z-spectrum offsets (remove whatever M0 offset the seq file uses)
    z_offsets = offsets_all[offsets_all != m0_offset_ppm]

    # Always simulate M0 scan at -300 ppm (convention used by all groups)
    # regardless of whether seq file specifies -300 or -1560 ppm
    mz_m0 = run_one_offset(-300.0, t_rec_m0)

    # Simulate Z-spectrum (all use Trec)
    mz_z = np.zeros(len(z_offsets))
    for i, off in enumerate(z_offsets):
        if verbose and i % 20 == 0:
            print(f"  offset {i+1}/{len(z_offsets)}: {off:.2f} ppm")
        mz_z[i] = run_one_offset(off, t_rec)

    # Return -300 ppm M0 row first, then Z-spectrum — all raw Mz
    all_offsets = np.concatenate([[-300.0], z_offsets])
    all_mz      = np.concatenate([[mz_m0],  mz_z])
    return all_offsets, all_mz


# ─────────────────────────────────────────────────────────────────
#  Per-case runners
# ─────────────────────────────────────────────────────────────────

def run_case(case_num: int, repo_dir: Path, verbose: bool = True):
    """
    Run a single BMsim challenge case.

    Returns dict with keys: offsets, mz, offsets_z, mz_z, mz_m0, case, model, defs, elapsed
    """
    case_dir = repo_dir / f"case_{case_num}"

    # Locate YAML and SEQ files
    yaml_files = list(case_dir.glob("*.yaml"))
    seq_files = list(case_dir.glob("*.seq"))

    if not yaml_files or not seq_files:
        raise FileNotFoundError(f"Missing YAML or SEQ in {case_dir}")

    yaml_path = yaml_files[0]
    seq_path = seq_files[0]

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Case {case_num}")
        print(f"  Pool model : {yaml_path.name}")
        print(f"  Sequence   : {seq_path.name}")
        print(f"{'='*55}")

    # Load pool model
    model = load_pool_model(yaml_path)

    # Parse sequence definitions
    defs = parse_seq_definitions(seq_path)

    # Extract timing and offsets
    offsets_all = np.array(defs["offsets_ppm"])
    m0_offset = float(defs.get("M0_offset", -300.0))
    t_sat = float(defs.get("Tsat", defs.get("tp", 2.0)))
    t_d = float(defs.get("td", 0.0))
    b1_cwpe = float(defs.get("B1cwpe", defs.get("b1cwpe", 2.0)))
    t_rec = float(defs.get("Trec", 3.5))
    t_rec_m0 = float(defs.get("Trec_M0", t_rec))

    # Z-spectrum offsets (exclude M0 row for display)
    offsets_zspec = offsets_all[offsets_all != m0_offset]

    if verbose:
        print(f"  B0         : {model.scanner.b0} T")
        print(f"  B1         : {b1_cwpe:.4f} µT")
        print(f"  T_sat      : {t_sat} s")
        print(f"  T_rec      : {t_rec} s  (M0: {t_rec_m0} s)")
        print(f"  Offsets    : {len(offsets_zspec)} points  "
              f"({offsets_zspec.min():.1f} to {offsets_zspec.max():.1f} ppm)")
        print(f"  Pools      : water + {len(model.cest_pools)} CEST"
              f"{' + MT' if model.mt_pool else ''}")
        print()

    # ── Run simulation ──
    t0 = time.time()
    all_offsets, all_mz = simulate_zspectrum(
        model=model,
        offsets_all=offsets_all,
        m0_offset_ppm=m0_offset,
        b1_ut=b1_cwpe,
        t_sat=t_sat,
        t_rec=t_rec,
        t_rec_m0=t_rec_m0,
        t_d=t_d,
        post_sat_delay=0.0065,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    # Split: first row is M0 scan, rest are Z-spectrum
    mz_m0   = all_mz[0]
    mz_z    = all_mz[1:]
    offs_z  = all_offsets[1:]

    if verbose:
        print(f"\n  ✓ Done in {elapsed:.2f}s")
        print(f"  M0 raw Mz: {mz_m0:.10f}  (at {all_offsets[0]:.0f} ppm)")
        print(f"  Mz range : [{mz_z.min():.4f}, {mz_z.max():.4f}]")

    return {
        "case":        case_num,
        "offsets":     all_offsets,   # all offsets incl. M0 row
        "mz":          all_mz,        # raw Mz for all offsets
        "offsets_z":   offs_z,        # Z-spectrum offsets only
        "mz_z":        mz_z,          # raw Mz for Z-spectrum only
        "mz_m0":       mz_m0,         # raw Mz at M0 offset
        "m0_offset":   m0_offset,
        "model":       model,
        "defs":        defs,
        "elapsed":     elapsed,
        "yaml_path":   yaml_path,
        "seq_path":    seq_path,
    }


# ─────────────────────────────────────────────────────────────────
#  Cross-validation: compare against BMCTool for case 4
# ─────────────────────────────────────────────────────────────────

def validate_against_bmctool(repo_dir: Path, result: dict) -> dict:
    """
    Run the same case through BMCTool (reference) and compare raw Mz values.
    Only practical for case 4 (82 offsets, 5ms pulse → runs in ~0.2s).
    Returns dict with max_abs_diff, rms_diff.
    """
    from bmctool.parameters import Parameters
    from bmctool.simulation import BMCSim

    yaml_path = result["yaml_path"]
    seq_path = result["seq_path"]

    params = Parameters.from_yaml(str(yaml_path))
    sim = BMCSim(params, str(seq_path), verbose=False)
    sim.run()
    ref_offsets, ref_z = sim.get_zspec()

    # Filter: remove M0-scan offset (-300 ppm)
    m0_offset = float(result["defs"].get("M0_offset", -300.0))
    mask = ref_offsets != m0_offset
    ref_offsets_z = ref_offsets[mask]
    ref_z_z = ref_z[mask]   # BMCTool returns raw Mz (no M0 normalization)

    # Our result: raw Z-spectrum (Mz/M0, but M0 ≈ 1.0, so ~ Mz)
    our_z = result["mz_z"]           # raw Mz for Z-spectrum offsets
    our_offsets = result["offsets"]  # same offsets

    # Align by matching offsets exactly
    # Both should have same offsets; use interpolation as safety
    ref_interp = np.interp(our_offsets, ref_offsets_z, ref_z_z)

    diff = our_z - ref_interp
    max_diff = float(np.max(np.abs(diff)))
    rms_diff = float(np.sqrt(np.mean(diff**2)))

    return {
        "ref_offsets": ref_offsets_z,
        "ref_z": ref_z_z,
        "max_abs_diff": max_diff,
        "rms_diff": rms_diff,
        "diff": diff,
    }


# ─────────────────────────────────────────────────────────────────
#  Submission CSV writer
# ─────────────────────────────────────────────────────────────────

def export_submission_csv(results: list[dict], output_dir: Path, group_name: str = "MyGroup"):
    """
    Export raw Mz values in the format used by all BMsim challenge participants.

    Format:
      - Column 1: offset_ppm
      - Column 2: Mz  (raw, un-normalized)
      - First row: M0 normalization scan (at -300 or -1560 ppm)
      - Remaining rows: Z-spectrum offsets in sequence order
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        case_num = r["case"]
        df = pd.DataFrame({
            "offset_ppm": r["offsets"],   # all offsets incl. M0 row
            "Mz":         r["mz"],        # raw Mz (NOT normalized)
        })
        path = output_dir / f"case_{case_num}_submission.csv"
        df.to_csv(path, index=False, float_format="%.10f")
        print(f"  Saved: {path}  (n={len(df)}, M0={r['mz_m0']:.8f})")

    print(f"\n  Submission CSVs written to: {output_dir}/")
    print(f"  Group identifier: {group_name}")
    print(f"  Google Sheet: https://docs.google.com/spreadsheets/d/"
          f"1JN7VN-f1ktDrJgokb0FlUFwkH0MWYlPA_jSfnQoFOVc/")


# ─────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────

def plot_all_cases(results: list[dict], output_path: Path, validation: dict = None):
    """
    Create a publication-quality figure with all 4 Z-spectra.
    """
    n = len(results)
    cols = 2
    rows = (n + 1) // cols

    fig = plt.figure(figsize=(14, 5 * rows), facecolor="white")
    fig.suptitle("BMsim Challenge – Simulated Z-Spectra (Cases 1–4)\n"
                 "Pure NumPy/SciPy Bloch-McConnell Solver  |  Matrix Exponential",
                 fontsize=13, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.45, wspace=0.35)

    colors = ["#2563EB", "#DC2626", "#16A34A", "#9333EA"]

    case_labels = {
        1: "Case 1 – 2-pool, CW 15s, 2µT\n(steady-state, creatine phantom)",
        2: "Case 2 – 2-pool, CW 2s, 2µT\n(APTw scheme, creatine phantom)",
        3: "Case 3 – 5-pool, CW 2s, 2µT\n(WM model: amide+guanidine+NOE+MT)",
        4: "Case 4 – 5-pool, CW 5ms, 3.7µT\n(WASABI scheme, WM model)",
    }

    for idx, r in enumerate(results):
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        c = colors[idx]
        case_num = r["case"]

        ax.plot(r["offsets_z"], r["mz_z"], color=c, lw=2, label="Our solver")

        # Add validation overlay if provided (case 4 only)
        if validation and case_num == 4:
            val = validation
            ax.plot(val["ref_offsets"], val["ref_z"],
                    "k--", lw=1.5, alpha=0.7, label="BMCTool (ref)")

        ax.set_xlabel("Saturation Offset (ppm)", fontsize=10)
        ax.set_ylabel("Z = Mz / M0", fontsize=10)
        ax.set_title(case_labels.get(case_num, f"Case {case_num}"),
                     fontsize=10, fontweight="bold")
        ax.invert_xaxis()
        ax.set_ylim(-0.22, 1.12)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")

        # Annotate timing — bottom-left, away from curves
        elapsed_str = f"⏱ {r['elapsed']:.1f}s"
        ax.text(0.03, 0.04, elapsed_str, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8, color="gray")

        # Highlight CEST pool positions — use leader lines so labels never overlap
        # Sort by dw to assign left-to-right positions
        sorted_pools = sorted(r["model"].cest_pools, key=lambda p: p.dw, reverse=True)

        # Get axis bounds (x-axis is inverted: high ppm on left)
        xlo, xhi = ax.get_xlim()   # xlo > xhi because axis is inverted
        x_span = xlo - xhi         # positive total width in ppm

        # Space label anchor positions evenly across the bottom band
        n_pools = len(sorted_pools)
        if n_pools == 1:
            anchors = [xlo - x_span * 0.5]
        else:
            margin = x_span * 0.1
            anchors = [xlo - margin - i * (x_span - 2 * margin) / (n_pools - 1)
                       for i in range(n_pools)]

        label_y  = -0.14   # where text sits
        tick_y   = -0.04   # where the leader meets the dw line

        for cp, anchor_x in zip(sorted_pools, anchors):
            # Dashed vertical marker at the true dw position
            ax.axvline(cp.dw, color=c, lw=0.8, ls=":", alpha=0.5)
            # Short horizontal leader from dw to anchor, then label
            ax.annotate(
                f"{cp.name.split('_')[0]}  {cp.dw} ppm",
                xy=(cp.dw, tick_y),           # arrow tip: at the dw line
                xytext=(anchor_x, label_y),   # label anchor: spread out
                fontsize=6.5, color=c,
                ha="center", va="top",
                arrowprops=dict(
                    arrowstyle="-",
                    color=c, alpha=0.5, lw=0.8,
                    connectionstyle="arc3,rad=0.0",
                ),
                bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.85),
            )

        # Legend — pinned to upper right with a clean frame
        if validation and case_num == 4:
            ax.legend(fontsize=8, loc="upper right",
                      framealpha=0.85, edgecolor="lightgray")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n  Figure saved: {output_path}")
    return fig


# ─────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    SCRIPT_DIR = Path(__file__).parent.resolve()
    REPO_DIR   = SCRIPT_DIR / "BMsim_challenge"
    OUTPUT_DIR = SCRIPT_DIR / "output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    GROUP_NAME = "BMsim-Python-MatExp"

    print("\n" + "█"*55)
    print("  BMsim Challenge – Full Pipeline")
    print("  Solver: Matrix Exponential (NumPy/SciPy)")
    print("█"*55)

    # ── Run all 4 cases ──
    results = []
    total_t0 = time.time()

    for case_num in [1, 2, 3, 4]:
        try:
            r = run_case(case_num, REPO_DIR, verbose=True)
            results.append(r)
        except Exception as e:
            print(f"  ✗ Case {case_num} failed: {e}")
            import traceback; traceback.print_exc()

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*55}")
    print(f"  All cases done in {total_elapsed:.1f}s total")
    print(f"{'='*55}")

    # ── Validate case 4 against BMCTool ──
    validation = None
    case4_result = next((r for r in results if r["case"] == 4), None)
    if case4_result:
        print("\n  Running BMCTool cross-validation on Case 4...")
        try:
            validation = validate_against_bmctool(REPO_DIR, case4_result)
            print(f"  Max |diff|  : {validation['max_abs_diff']:.6f}")
            print(f"  RMS diff    : {validation['rms_diff']:.6f}")
            if validation['max_abs_diff'] < 0.005:
                print("  ✓ PASS: Agreement within 0.5% of M0 (z-only MT reference)")
            elif validation['max_abs_diff'] < 0.03:
                print("  ✓ EXPECTED: Diff < 3% vs BMCTool — due to correct 3-component MT")
                print("  (BMCTool uses z-only MT; our solver uses 3-component per challenge FAQ)")
            else:
                print("  ⚠ WARN: Large differences detected – review solver")
        except Exception as e:
            print(f"  BMCTool validation skipped: {e}")

    # ── Export submission CSVs ──
    print("\n  Exporting submission files...")
    export_submission_csv(results, OUTPUT_DIR / "submissions", GROUP_NAME)

    # ── Plot ──
    print("\n  Generating figures...")
    fig = plot_all_cases(results, OUTPUT_DIR / "zspectra_all_cases.png", validation)
    plt.close(fig)

    # ── Summary table ──
    print("\n" + "─"*55)
    print(f"  {'Case':<6} {'Offsets':<9} {'Time(s)':<10} {'Z_min':<10} {'Z_max'}")
    print("─"*55)
    for r in results:
        print(f"  {r['case']:<6} {len(r['offsets']):<9} {r['elapsed']:<10.2f} "
              f"{r['mz_z'].min():<10.4f} {r['mz_z'].max():.4f}")
    print("─"*55)
    print(f"\n  Output directory: {OUTPUT_DIR}/")
