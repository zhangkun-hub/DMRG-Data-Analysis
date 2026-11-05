"""Generate all thesis figures for the honeycomb t-J DMRG paper.

Run from `DMRG data visualization/` under Calculation env:
    python -X utf8 generate_thesis_figures.py
"""
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

OUTDIR = Path("output/thesis_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

_original_show = plt.show
plt.show = lambda *a, **kw: None

def save_current(name):
    path = OUTDIR / f"{name}.png"
    plt.gcf().savefig(path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved: {path}")

N = 200
DOPING = "d1_100"
DATA_ROOT = Path("Data/Data_N200/t_J_3")
MEASURE_DIR = DATA_ROOT / DOPING / "1500_measure_phys_qty" / "measure_49"
SC_DIR = DATA_ROOT / DOPING / "1500_measure_super_corr" / "super_corr_49_50"

# ── imports for physical quantities ────────────────────────────────────
from Physical_Quantity.data_records.Physical_Data_Record import (
    DensityRecord, CorrelationRecord, SuperCorrRecord)
from Physical_Quantity.physical_analysis.Density_Process import DensityProcessor
from Physical_Quantity.physical_analysis.Correlation_Process import CorrelationProcessor
from Physical_Quantity.physical_analysis.Super_Correlation_Process import SuperCorrProcessor
from Physical_Quantity.physical_analysis.Super_Correlation_Analyzer import SuperCorrDecayAnalyzer
from Physical_Quantity.physical_analysis.Correlation_Decay_Analyzer import CorrelationDecayAnalyzer
from Physical_Quantity.physical_analysis.Energy_Process import EnergyRecord, EnergyProcessor

# ── Fig 1: Electron density profile ──────────────────────────────────
def fig1_density():
    print("[Fig 1] Electron density profile")
    text = (MEASURE_DIR / "nf.json").read_text(encoding='utf-8')
    recs = DensityRecord.extract_data(text, N=N, a=1.0)
    DensityProcessor.Line_Chart_Visualization(recs)
    save_current("fig1_density_d1_100")

# ── Fig 2: Spin correlation ⟨S_i·S_j⟩ ────────────────────────────────
def _extract_spin(base, ver, src_off, max_off):
    sp = int(N / 8 / 4) * 8 + src_off
    fp_max = int(N / 4 * 3) + max_off
    seq = range(sp, fp_max + 1, 4)
    SzSz = CorrelationRecord.extract_data(
        (base / f"sz{ver}sz.json").read_text('utf-8'), seq, N)
    SuSd = CorrelationRecord.extract_data(
        (base / f"sp{ver}sm.json").read_text('utf-8'), seq, N)
    SdSu = CorrelationRecord.extract_data(
        (base / f"sm{ver}sp.json").read_text('utf-8'), seq, N)
    return CorrelationProcessor.Combined_spin_correlation(SzSz, SuSd, SdSu)

def fig2_spin_correlation():
    print("[Fig 2] Spin correlation")
    base_49 = MEASURE_DIR  # measure_49
    base_50 = DATA_ROOT / DOPING / "1500_measure_phys_qty" / "measure_50"
    SiSj_A = _extract_spin(base_49, 49, src_off=1, max_off=-3)
    SiSj_B = _extract_spin(base_50, 50, src_off=2, max_off=-2)
    CorrelationProcessor.Spin_Corr_Visualization(
        {'A sublattice': SiSj_A, 'B sublattice': SiSj_B})
    save_current("fig2_spin_corr_d1_100")

# ── Fig 3: SC pairing correlation (three bond directions) ────────────
def _load_sc_records(sc_dir, refbond_str):
    records = {}
    for bond in 'ABC':
        for ch in 'abcd':
            key = f'scs{bond}{ch}'
            fname = f'{key}_fix_refbond_{refbond_str}.json'
            records[key] = SuperCorrRecord.extract_data(
                (sc_dir / fname).read_text('utf-8'), N=N, a=1.0)
    return records

def _combine_sc(records):
    bond_labels = {
        'A': "A key (blue bond)",
        'B': "B key (green bond)",
        'C': "C key (orange bond)",
    }
    sc_dict = {}
    for b, label in bond_labels.items():
        sc_dict[label] = SuperCorrProcessor.Combined_super_correlation(
            *[records[f'scs{b}{c}'] for c in 'abcd'])
    return sc_dict, bond_labels

def fig3_sc_correlation():
    print("[Fig 3] SC pairing correlation (3 bond directions)")
    records = _load_sc_records(SC_DIR, "49_50")
    sc_dict, _ = _combine_sc(records)
    SuperCorrProcessor.Super_Corr_Visualization_three(sc_dict)
    save_current("fig3_sc_corr_d1_100_ref49_50")
# ── Fig 4: SC decay analysis (2×2 standard panel per direction) ──────
def fig4_sc_decay():
    print("[Fig 4] SC decay analysis")
    records = _load_sc_records(SC_DIR, "49_50")
    sc_dict, bond_labels = _combine_sc(records)
    for bond_char, label in bond_labels.items():
        combined_recs = sc_dict[label]
        prep = SuperCorrDecayAnalyzer.group_by_distance(combined_recs)
        result = SuperCorrDecayAnalyzer.analyze(combined_recs)
        out = str(OUTDIR / f"fig4_sc_decay_d1_100_49_50_{bond_char}.png")
        CorrelationDecayAnalyzer.Decay_Visualization(
            result, r=prep.r, C=prep.C, show=False, save_path=out,
            visualization_mode='standard',
            title=f'{bond_char} direction (refbond 49_50)')
        print(f"  Saved: {out}")

# ── Fig 5: Spin structure factor S(q) heatmap ────────────────────────
def fig5_spin_structure_factor():
    print("[Fig 5] Spin structure factor S(q)")
    from Physical_Quantity.physical_analysis.Spin_Structure_Factor import (
        SpinStructureFactorEstimator, xc_momentum_grid)
    from Physical_Quantity.physical_analysis.Spin_Structure_Factor_Analyzer import (
        SpinStructureFactorPeakAnalyzer)
    from Physical_Quantity.evidence_integration.Analysis_Integration import (
        structure_factor_report, peak_analysis_report)

    base_49 = MEASURE_DIR
    base_50 = DATA_ROOT / DOPING / "1500_measure_phys_qty" / "measure_50"
    seq_a = range(int(N/8/4)*8+1, int(N/4*3)-3+1, 4)
    seq_b = range(int(N/8/4)*8+2, int(N/4*3)-2+1, 4)

    SiSj_A = _extract_spin(base_49, 49, src_off=1, max_off=-3)
    SiSj_B = _extract_spin(base_50, 50, src_off=2, max_off=-2)
    combined_records = SiSj_A + SiSj_B

    q_list, _meta = xc_momentum_grid(N=N, n_qx=41, n_qy=4, a=1.0)
    sf_report = structure_factor_report(
        SpinStructureFactorEstimator.estimate_spin_dot_from_records,
        combined_records, q_list, N, a=1.0)
    peak_report = peak_analysis_report(sf_report.payload)

    out = str(OUTDIR / "fig5_spin_sf_d1_100.png")
    SpinStructureFactorPeakAnalyzer.plot_peaks(
        sf_report.payload, peak_report.payload,
        save_path=out, show=False)
    print(f"  Saved: {out}")
# ── Fig 6: Energy extrapolation E∞ vs 1/N ────────────────────────────
ENERGY_DOPINGS = {
    'd0': 0.0, 'd1_10': 0.1, 'd1_20': 0.05,
    'd1_40': 0.025, 'd1_80': 0.0125, 'd3_40': 0.075,
}
J_VALUE = 1.0 / 3.0

def fig6_energy_extrapolation():
    print("[Fig 6] Energy extrapolation E∞ vs 1/N")
    e_inf_results = {}
    for doping, delta in ENERGY_DOPINGS.items():
        records = []
        for n in (80, 160, 240, 320, 400):
            log_dir = Path(f"Data/Data_Energy/t_J_3/{doping}/N{n}")
            logs = sorted(log_dir.glob("slurm-*.log"))
            if not logs:
                print(f"  WARNING: no log for {doping}/N{n}, skipping")
                continue
            text = logs[0].read_text(encoding='utf-8', errors='replace')
            rec = EnergyRecord.extract_data(text, N=n, J=J_VALUE)
            records.append(rec)
        if len(records) >= 3:
            e_inf = EnergyProcessor.extrapolate(records, power=1)
            save_current(f"fig6_energy_extrap_{doping}")
            e_inf_results[doping] = (delta, e_inf)
        else:
            print(f"  WARNING: {doping} has <3 records, skipping extrapolation")
    return e_inf_results

# ── Fig 7: E∞ vs doping δ ────────────────────────────────────────────
def fig7_energy_vs_doping(e_inf_results):
    print("[Fig 7] E∞ vs doping δ")
    if len(e_inf_results) < 2:
        print("  WARNING: not enough data for doping dependence fit")
        return
    sorted_items = sorted(e_inf_results.values(), key=lambda x: x[0])
    deltas = [x[0] for x in sorted_items]
    e_infs = [x[1] for x in sorted_items]
    EnergyProcessor.fit_doping_dependence(deltas, e_infs)
    save_current("fig7_energy_vs_doping")

# ── Fig 8: K_sc(δ) evolution ─────────────────────────────────────────
SC_DOPING_DELTA = {
    'd1_10': 0.10, 'd1_20': 0.05, 'd1_100': 0.01, 'd2_25': 0.08,
}

def fig8_ksc_vs_doping():
    print("[Fig 8] K_sc(δ) evolution")
    csv_path = Path("output/sc_exponent_summary/sc_exponent_summary.csv")
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping")
        return
    df = pd.read_csv(csv_path)
    df['delta'] = df['doping'].map(SC_DOPING_DELTA)
    markers = {'49_50': 'o', '50_51': 's', '50_54': 'D'}
    colors = {'A': 'tab:blue', 'B': 'tab:green', 'C': 'tab:orange'}

    df_K = df[df['exponent_type'] == 'K'].copy()
    df_xi = df[df['exponent_type'] == 'xi'].copy()
    has_K = len(df_K) > 0
    has_xi = len(df_xi) > 0
    n_panels = has_K + has_xi
    if n_panels == 0:
        print("  WARNING: no K or xi data found")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), squeeze=False)
    ax_idx = 0
    if has_K:
        ax = axes[0, ax_idx]
        for (rb, dirn), grp in df_K.groupby(['refbond', 'direction']):
            grp_s = grp.sort_values('delta')
            ax.errorbar(grp_s['delta'], grp_s['exponent_value'],
                        yerr=grp_s['exponent_error'],
                        marker=markers.get(rb, 'x'), color=colors.get(dirn, 'gray'),
                        label=f'ref {rb}, {dirn}', capsize=3, ls='--', alpha=0.8)
        ax.set_xlabel(r'Hole doping $\delta$', fontsize=13)
        ax.set_ylabel(r'$K_{sc}$', fontsize=13)
        ax.set_title('Power-law exponent $K_{sc}$', fontsize=14)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax_idx += 1
    if has_xi:
        ax = axes[0, ax_idx]
        for (rb, dirn), grp in df_xi.groupby(['refbond', 'direction']):
            grp_s = grp.sort_values('delta')
            ax.errorbar(grp_s['delta'], grp_s['exponent_value'],
                        yerr=grp_s['exponent_error'],
                        marker=markers.get(rb, 'x'), color=colors.get(dirn, 'gray'),
                        label=f'ref {rb}, {dirn}', capsize=3, ls='--', alpha=0.8)
        ax.set_xlabel(r'Hole doping $\delta$', fontsize=13)
        ax.set_ylabel(r'$\xi$', fontsize=13)
        ax.set_title(r'Correlation length $\xi$', fontsize=14)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_current("fig8_Ksc_vs_doping")

# ── Fig 9: Convergence diagnostics ───────────────────────────────────
def fig9_convergence():
    print("[Fig 9] Convergence diagnostics")
    import subprocess
    conv_dir = OUTDIR / "convergence"
    cmd = [
        sys.executable, "-X", "utf8", "Phase_Analysis_Report.py",
        "--root", "Data/Data_N200", "--t-j", "t_J_3",
        "--doping", "d1_100", "--N", "200", "--bond-dim", "1500",
        "--convergence-dims", "1500,5000",
        "--outdir", str(conv_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                            errors='replace', timeout=300)
    if result.returncode == 0:
        plots_dir = conv_dir / "plots"
        if plots_dir.exists():
            for f in sorted(plots_dir.glob("*.png")):
                print(f"  Saved: {f}")
        else:
            print("  WARNING: no plots directory created")
    else:
        print(f"  WARNING: convergence report failed (exit {result.returncode})")
        print(f"  stderr: {result.stderr[-200:]}" if result.stderr else "")

if __name__ == "__main__":
    print("=== Thesis Figure Generation ===")
    print(f"Output: {OUTDIR.resolve()}\n")

    fig1_density()
    fig2_spin_correlation()
    fig3_sc_correlation()
    fig4_sc_decay()
    fig5_spin_structure_factor()
    e_inf_results = fig6_energy_extrapolation()
    fig7_energy_vs_doping(e_inf_results)
    fig8_ksc_vs_doping()
    fig9_convergence()

    pngs = list(OUTDIR.rglob("*.png"))
    print(f"\n=== Complete: {len(pngs)} figures generated ===")
    for p in sorted(pngs):
        print(f"  {p}")
