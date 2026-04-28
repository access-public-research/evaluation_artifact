import math
from pathlib import Path
import pandas as pd


def mean_ci(vals):
    vals = [float(v) for v in vals]
    n = len(vals)
    if n == 0:
        return float('nan'), float('nan')
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(max(var, 0.0))
    t_map = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045}
    t = t_map.get(n, 1.96)
    return m, t * sd / math.sqrt(n)


def fmt(v, digits=3):
    if pd.isna(v):
        return 'n/a'
    return f'{float(v):.{digits}f}'


def fmt_pm(m, ci, digits=3):
    return f'{float(m):.{digits}f} $\\pm$ {float(ci):.{digits}f}'


def write_fixed_partition_table(root: Path):
    metrics = root / 'replication_rcg' / 'artifacts' / 'metrics'
    out = root / 'paper' / 'neurips2026_selection_risk' / 'tables' / 'table_fixed_partition_controls.tex'

    ce = pd.read_csv(metrics / 'celeba_effect_size_v20celeba_fixedconf_10s_20260308.csv')
    ce_test = pd.read_csv(metrics / 'celeba_test_wg_selected_summary_v20celeba_fixedconf_10s_20260308.csv')
    ce = ce.merge(ce_test[['regime', 'test_oracle_wg_acc_mean', 'test_oracle_wg_acc_ci']], on='regime', how='left')
    ce_base = ce.loc[ce['regime'] == 'rcgdro'].iloc[0]

    cam = pd.read_csv(metrics / 'camelyon17_effect_size_v20cam_fixedconf_10s_20260308.csv')
    cam_perf = pd.read_csv(metrics / 'camelyon17_resnet50_domain_acc_v20cam_fixedconf_10s_20260308.csv')
    cam_perf_sum = []
    for regime, sub in cam_perf.groupby('regime'):
        m, ci = mean_ci(sub['test_hosp_2_acc'])
        cam_perf_sum.append({'regime': regime, 'test_hosp_2_acc_mean': m, 'test_hosp_2_acc_ci': ci})
    cam_perf_sum = pd.DataFrame(cam_perf_sum)
    cam = cam.merge(cam_perf_sum, on='regime', how='left')
    cam_base = cam.loc[cam['regime'] == 'rcgdro'].iloc[0]

    rows = []
    reg_display = {'rcgdro': 'rcgdro', 'rcgdro_softclip_p95_a10': 'P95', 'rcgdro_softclip_p97_a10': 'P97', 'rcgdro_softclip_p99_a10': 'P99', 'rcgdro_softclip_p95_a10_cam': 'P95', 'rcgdro_softclip_p97_a10_cam': 'P97', 'rcgdro_softclip_p99_a10_cam': 'P99'}
    for _, r in ce.iterrows():
        delta_proxy = 0.0 if r['regime'] == 'rcgdro' else r['proxy_worst_loss_clip_mean'] - ce_base['proxy_worst_loss_mean']
        delta_tail = r['tail_worst_cvar_mean'] - ce_base['tail_worst_cvar_mean']
        delta_perf = r['test_oracle_wg_acc_mean'] - ce_base['test_oracle_wg_acc_mean']
        rows.append(['CelebA', reg_display[r['regime']], fmt(r['frac_clipped_val_mean']) if pd.notna(r['frac_clipped_val_mean']) else '0.000', fmt(delta_proxy), f"{fmt(r['tail_worst_cvar_mean'],2)} ({delta_tail:+.2f})", f"{fmt(r['test_oracle_wg_acc_mean'])} ({delta_perf:+.3f})"])
    for _, r in cam.iterrows():
        delta_proxy = 0.0 if r['regime'] == 'rcgdro' else r['proxy_worst_loss_clip_mean'] - cam_base['proxy_worst_loss_mean']
        delta_tail = r['tail_worst_cvar_mean'] - cam_base['tail_worst_cvar_mean']
        delta_perf = r['test_hosp_2_acc_mean'] - cam_base['test_hosp_2_acc_mean']
        rows.append(['Camelyon17', reg_display[r['regime']], fmt(r['frac_clipped_val_mean']) if pd.notna(r['frac_clipped_val_mean']) else '0.000', fmt(delta_proxy), f"{fmt(r['tail_worst_cvar_mean'],2)} ({delta_tail:+.2f})", f"{fmt(r['test_hosp_2_acc_mean'])} ({delta_perf:+.3f})"])

    lines = [r'\begin{tabular}{llcccc}', r'  \toprule', r'  Dataset & Regime & FracClip & $\Delta$Proxy$\downarrow$ & Tail CVaR$\downarrow$ ($\Delta$) & Perf$\uparrow$ ($\Delta$) \\', r'  \midrule']
    for i, row in enumerate(rows):
        if i == 4:
            lines.append(r'  \midrule')
        lines.append('  ' + ' & '.join(row) + r' \\')
    lines.extend([r'  \bottomrule', r'\end{tabular}'])
    out.write_text('\n'.join(lines) + '\n')


def write_dense_cam_table(root: Path):
    metrics = root / 'replication_rcg' / 'artifacts' / 'metrics'
    out = root / 'paper' / 'neurips2026_selection_risk' / 'tables' / 'table_camelyon_dense_sweep_20260308.tex'

    eff = pd.read_csv(metrics / 'camelyon17_effect_size_v20cam_dense_10s_20260308.csv')
    perf = pd.read_csv(metrics / 'camelyon17_resnet50_domain_acc_v20cam_dense_10s_20260308.csv')
    perf_sum = []
    for regime, sub in perf.groupby('regime'):
        m, ci = mean_ci(sub['test_hosp_2_acc'])
        perf_sum.append({'regime': regime, 'test_hosp_2_acc_mean': m, 'test_hosp_2_acc_ci': ci})
    perf_sum = pd.DataFrame(perf_sum)
    eff = eff.merge(perf_sum, on='regime', how='left')
    base = eff.loc[eff['regime'] == 'rcgdro'].iloc[0]
    reg_order = ['rcgdro', 'rcgdro_softclip_p95_a10_cam', 'rcgdro_softclip_p96_a10_cam', 'rcgdro_softclip_p97_a10_cam', 'rcgdro_softclip_p98_a10_cam', 'rcgdro_softclip_p99_a10_cam']
    reg_display = {'rcgdro': 'rcgdro', 'rcgdro_softclip_p95_a10_cam': 'P95', 'rcgdro_softclip_p96_a10_cam': 'P96', 'rcgdro_softclip_p97_a10_cam': 'P97', 'rcgdro_softclip_p98_a10_cam': 'P98', 'rcgdro_softclip_p99_a10_cam': 'P99'}

    lines = [r'\begin{tabular}{lccccc}', r'  \toprule', r'  Regime & FracClip & Proxy$\downarrow$ & Tail CVaR$\downarrow$ ($\Delta$) & Test-hosp2$\uparrow$ ($\Delta$) & Note \\', r'  \midrule']
    for reg in reg_order:
        r = eff.loc[eff['regime'] == reg].iloc[0]
        delta_tail = r['tail_worst_cvar_mean'] - base['tail_worst_cvar_mean']
        delta_perf = r['test_hosp_2_acc_mean'] - base['test_hosp_2_acc_mean']
        note = 'baseline' if reg == 'rcgdro' else ('tail peak' if reg == 'rcgdro_softclip_p96_a10_cam' else ('tail onset rec.' if reg == 'rcgdro_softclip_p99_a10_cam' else ''))
        frac = '0.000' if reg == 'rcgdro' else fmt(r['frac_clipped_val_mean'])
        proxy = fmt(r['proxy_worst_loss_clip_mean']) if reg != 'rcgdro' else fmt(r['proxy_worst_loss_mean'])
        lines.append('  ' + ' & '.join([reg_display[reg], frac, proxy, f"{fmt(r['tail_worst_cvar_mean'],2)} ({delta_tail:+.2f})", f"{fmt(r['test_hosp_2_acc_mean'])} ({delta_perf:+.3f})", note]) + r' \\')
    lines.extend([r'  \bottomrule', r'\end{tabular}'])
    out.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[3]
    write_fixed_partition_table(root)
    write_dense_cam_table(root)
