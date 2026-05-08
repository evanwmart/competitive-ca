#!/usr/bin/env python3
"""
tau_bootstrap.py -- Bootstrap confidence intervals on the tau(inf) extrapolation.

Residual-resampling bootstrap for the linear fit of tau vs 1/L.
Reports 3-point (L=64,128,256) and 4-point (including L=512) fits.

Usage:
    venv/bin/python3 analysis/tau_bootstrap.py
"""

import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── style (from figures.py) ─────────────────────────────────────────────────

C_RAND  = '#2166ac'
C_L512  = '#7b3294'   # distinct colour for 4-point fit

FONT_TITLE  = 11
FONT_LABEL  = 11
FONT_TICK   = 9
FONT_LEGEND = 8.5
FONT_ANNOT  = 8

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         FONT_TICK,
    'axes.labelsize':    FONT_LABEL,
    'axes.titlesize':    FONT_TITLE,
    'legend.fontsize':   FONT_LEGEND,
    'xtick.labelsize':   FONT_TICK,
    'ytick.labelsize':   FONT_TICK,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.15,
    'grid.linewidth':    0.4,
    'figure.dpi':        200,
})

ROOT   = Path(__file__).parent.parent
LOGDIR = ROOT / 'logs'
FIGDIR = ROOT / 'paper'

# ── extract tau values from logs ────────────────────────────────────────────

def read_tau(logfile, mr_target=5):
    """Parse tau for a given mr from a histogram log file."""
    text = logfile.read_text()
    for line in text.splitlines():
        if re.search(rf'mr=\s*{mr_target}\b', line):
            m_tau = re.search(r'[τt](?:au)?=([0-9.]+)', line)
            m_r2  = re.search(r'R[²2]=([0-9.]+)', line)
            if m_tau and m_r2:
                return float(m_tau.group(1)), float(m_r2.group(1))
    return None, None


logfiles = [
    (64,  LOGDIR / '2_tau_L64.log'),
    (128, LOGDIR / '2_tau_L128.log'),
    (256, LOGDIR / '2_tau_L256.log'),
    (512, LOGDIR / '2_tau_L512.log'),
]

Ls, taus, r2s = [], [], []
for L, path in logfiles:
    tau, r2 = read_tau(path)
    if tau is not None:
        Ls.append(L)
        taus.append(tau)
        r2s.append(r2)

Ls   = np.array(Ls)
taus = np.array(taus)
r2s  = np.array(r2s)

print("Extracted tau values (mr=5):")
for L, tau, r2 in zip(Ls, taus, r2s):
    print(f"  L={L:4d}  tau={tau:.3f}  R^2={r2:.4f}")
print()

# ── bootstrap machinery ────────────────────────────────────────────────────

N_BOOT = 10_000
rng = np.random.default_rng(2026)


def linear_fit(x, y):
    """Return (slope, intercept) for y = slope*x + intercept."""
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept


def residual_bootstrap(x, y, n_boot=N_BOOT):
    """
    Residual-resampling bootstrap for linear regression.
    Returns array of bootstrap intercepts (= tau(inf) estimates).
    """
    slope, intercept = linear_fit(x, y)
    y_hat = slope * x + intercept
    residuals = y - y_hat

    boot_intercepts = np.empty(n_boot)
    boot_slopes = np.empty(n_boot)
    for i in range(n_boot):
        resamp = rng.choice(residuals, size=len(residuals), replace=True)
        y_boot = y_hat + resamp
        s, b = linear_fit(x, y_boot)
        boot_intercepts[i] = b
        boot_slopes[i] = s
    return boot_intercepts, boot_slopes, slope, intercept


def report(label, x, y, ref_values):
    """Run bootstrap and print results."""
    intercepts, slopes, slope0, intercept0 = residual_bootstrap(x, y)
    ci_lo, ci_hi = np.percentile(intercepts, [2.5, 97.5])
    print(f"=== {label} ===")
    print(f"  Point estimate:  tau(inf) = {intercept0:.4f}")
    print(f"  Bootstrap mean:  {np.mean(intercepts):.4f}")
    print(f"  Bootstrap std:   {np.std(intercepts):.4f}")
    print(f"  95% CI:          [{ci_lo:.4f}, {ci_hi:.4f}]")
    for name, val in ref_values.items():
        inside = "YES" if ci_lo <= val <= ci_hi else "NO"
        print(f"  Contains {name} = {val:.4f}?  {inside}")
    print()
    return intercepts, slopes, slope0, intercept0


# ── run both fits ───────────────────────────────────────────────────────────

inv_L = 1.0 / Ls
ref = {'voter (tau=2.0)': 2.0, 'percolation (tau=187/91)': 187/91}

# 3-point: L=64, 128, 256
mask3 = Ls <= 256
int3, sl3, s0_3, b0_3 = report(
    "3-point fit (L=64, 128, 256)", inv_L[mask3], taus[mask3], ref)

# 4-point: L=64, 128, 256, 512
int4, sl4, s0_4, b0_4 = report(
    "4-point fit (L=64, 128, 256, 512)", inv_L, taus, ref)


# ── figure ──────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)

# Data points
ax.plot(inv_L, taus, 'o', color=C_RAND, ms=8, zorder=5,
        label=r'Measured $\tau$ (mr=5, $\mu=0.20$)')

# Annotations
offsets = {64: (18, 18), 128: (35, 20), 256: (18, 18), 512: (-20, 45)}
for L, tau, r2 in zip(Ls, taus, r2s):
    ox, oy = offsets.get(L, (18, 12))
    ax.annotate(f'L={L}, R$^2$={r2:.3f}', xy=(1/L, tau),
                xytext=(ox, oy), textcoords='offset points',
                fontsize=FONT_ANNOT,
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.6))

# Reference lines
ax.axhline(2.0, color='#666', ls='--', lw=1.2, zorder=1,
           label=r'$\tau=2.0$ (voter model)')
tau_perc = 187 / 91
ax.axhline(tau_perc, color='#aaa', ls=':', lw=1, zorder=1,
           label=r'$\tau=\frac{187}{91}\approx%.3f$ (2D percolation)' % tau_perc)

x_extrap = np.linspace(0, inv_L[0] * 1.05, 300)

# --- 3-point fit with CI band ---
ci3_lo, ci3_hi = np.percentile(int3, [2.5, 97.5])
sl3_lo, sl3_hi = np.percentile(sl3, [2.5, 97.5])

# Best fit line
ax.plot(x_extrap, s0_3 * x_extrap + b0_3, '-', color=C_RAND, lw=1.2,
        alpha=0.5, label=(r'3-pt fit: $\tau(\infty)=%.3f$  [%.3f, %.3f]'
                          % (b0_3, ci3_lo, ci3_hi)))

# CI band: envelope from bootstrap realisations
boot_lines_3 = sl3[:, None] * x_extrap[None, :] + int3[:, None]
lo3 = np.percentile(boot_lines_3, 2.5, axis=0)
hi3 = np.percentile(boot_lines_3, 97.5, axis=0)
ax.fill_between(x_extrap, lo3, hi3, color=C_RAND, alpha=0.12)

# Extrapolated point
ax.plot(0, b0_3, 'o', color=C_RAND, ms=9, zorder=6,
        markerfacecolor='white', markeredgewidth=1.5)

# --- 4-point fit with CI band ---
ci4_lo, ci4_hi = np.percentile(int4, [2.5, 97.5])

ax.plot(x_extrap, s0_4 * x_extrap + b0_4, '-', color=C_L512, lw=1.2,
        alpha=0.5, label=(r'4-pt fit: $\tau(\infty)=%.3f$  [%.3f, %.3f]'
                          % (b0_4, ci4_lo, ci4_hi)))

boot_lines_4 = sl4[:, None] * x_extrap[None, :] + int4[:, None]
lo4 = np.percentile(boot_lines_4, 2.5, axis=0)
hi4 = np.percentile(boot_lines_4, 97.5, axis=0)
ax.fill_between(x_extrap, lo4, hi4, color=C_L512, alpha=0.10)

ax.plot(0, b0_4, 's', color=C_L512, ms=9, zorder=6,
        markerfacecolor='white', markeredgewidth=1.5)

# Axes
ax.set_xlabel(r'Inverse system size  $1/L$')
ax.set_ylabel(r'Domain-size exponent  $\tau$')
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
ax.set_xlim(-0.002, inv_L[0] * 1.4)
ax.set_ylim(1.6, 2.5)
ax.legend(fontsize=FONT_LEGEND, loc='upper right')
ax.set_title(r'Finite-size extrapolation of $\tau$ with bootstrap 95\% CI'
             '\n'
             r'Fixed lattice, $\mu=0.20$, $r_{\min}=4$, 32 seeds per $L$')

# Save
FIGDIR.mkdir(exist_ok=True)
path = FIGDIR / 'fig_tau_bootstrap.pdf'
fig.savefig(path, bbox_inches='tight')
fig.savefig(path.with_suffix('.png'), bbox_inches='tight')
plt.close(fig)
print(f"Saved: {path}")
print(f"Saved: {path.with_suffix('.png')}")
