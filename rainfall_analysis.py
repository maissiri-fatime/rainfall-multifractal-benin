"""
=============================================================
STEP 1 — Load and Convert Excel to CSV
=============================================================
Author  : [Fatime CHOU]
Input   : rainfall_data.xlsx
Output  : rainfall_benin_clean.csv
=============================================================
"""

import pandas as pd
import numpy as np

# ── 1. Load the Excel file ────────────────────────────────────
print("Loading Excel file...")
df = pd.read_excel('rainfall_data.xlsx')

# ── 2. Convert the date column ────────────────────────────────
# Column 'Years' is in YYYYMMDD format (e.g. 19810101)
df['date'] = pd.to_datetime(df['Years'].astype(str), format='%Y%m%d')
df = df.drop(columns=['Years'])
df = df.set_index('date')

# ── 3. Replace -99.9 with NaN ─────────────────────────────────
# -99.9 is a sentinel code meaning missing data (common in meteorology)
df = df.replace(-99.9, np.nan)

# ── 4. Rename columns ─────────────────────────────────────────
df.columns = ['Bohicon', 'Cotonou', 'Parakou']

# ── 5. Save to CSV ────────────────────────────────────────────
df.to_csv('rainfall_benin_clean.csv')
print("File saved: rainfall_benin_clean.csv")

# ── 6. Quick summary ──────────────────────────────────────────
print("\n=== DATA SUMMARY ===")
print(f"Period : {df.index[0].date()} to {df.index[-1].date()}")
print(f"Rows   : {len(df)}")

print("\nMissing values per station:")
print(df.isnull().sum())

print("\nMissing values (%):")
print((df.isnull().sum() / len(df) * 100).round(2))

print("\nDescriptive statistics:")
print(df.describe().round(2))

# ── 7. Warning about Parakou ──────────────────────────────────
print("\n*** WARNING ***")
print("Parakou has 2782 missing values (18.1% of the series).")
print("The largest gap: 2740 consecutive missing days (July 2012 to Jan 2020).")
print("=> In all subsequent steps, Parakou will be restricted to 1981-2012.")
"""
=============================================================
STEP 2 — Stationarity Tests (ADF + KPSS)
=============================================================
Input   : rainfall_benin_clean.csv
Output  : stationarity_results.csv
          stationarity_analysis.png
=============================================================

THEORETICAL BACKGROUND:
- ADF test  : H0 = unit root (NON-stationary)
              If p < 0.05 => reject H0 => series is STATIONARY
- KPSS test : H0 = stationarity
              If p > 0.05 => fail to reject H0 => series is STATIONARY
- Both tests must agree to draw a firm conclusion.
- A stationary series is required before applying DFA and MF-DFA.
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load data ──────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('rainfall_benin_clean.csv', parse_dates=['date'], index_col='date')

# Parakou: restrict to 1981-2012 (large data gap 2012-2020)
stations = {
    'Bohicon' : df['Bohicon'].dropna(),
    'Cotonou' : df['Cotonou'].dropna(),
    'Parakou' : df['Parakou']['1981':'2012'].dropna()
}

# ── 2. Run tests ──────────────────────────────────────────────
print("\n" + "="*60)
print("       STATIONARITY TESTS")
print("="*60)

results = []

for name, series in stations.items():
    print(f"\n{'─'*55}")
    print(f"  Station : {name}   (N = {len(series)} days)")
    print(f"{'─'*55}")

    # -- ADF test ---------------------------------------------
    adf_stat, adf_p, _, _, adf_cv, _ = adfuller(series, autolag='AIC')
    adf_stationary = adf_p < 0.05

    print(f"\n  [ADF]  Statistic  = {adf_stat:.4f}")
    print(f"         p-value    = {adf_p:.6f}")
    print(f"         Crit. val. : 1%={adf_cv['1%']:.3f}  "
          f"5%={adf_cv['5%']:.3f}  10%={adf_cv['10%']:.3f}")
    print(f"         => {'STATIONARY' if adf_stationary else 'NON-STATIONARY'}")

    # -- KPSS test --------------------------------------------
    kpss_stat, kpss_p, _, kpss_cv = kpss(series, regression='c', nlags='auto')
    kpss_stationary = kpss_p > 0.05

    print(f"\n  [KPSS] Statistic  = {kpss_stat:.4f}")
    print(f"         p-value    = {kpss_p:.4f}")
    print(f"         Crit. val. : 1%={kpss_cv['1%']:.3f}  "
          f"5%={kpss_cv['5%']:.3f}  10%={kpss_cv['10%']:.3f}")
    print(f"         => {'STATIONARY' if kpss_stationary else 'NON-STATIONARY'}")

    # -- Conclusion -------------------------------------------
    if adf_stationary and kpss_stationary:
        conclusion = "STATIONARY (both tests agree)"
    elif adf_stationary and not kpss_stationary:
        conclusion = "CONTRADICTORY — possible trend present"
    else:
        conclusion = "NON-STATIONARY (both tests agree)"

    print(f"\n  CONCLUSION: {conclusion}")

    results.append({
        'Station'         : name,
        'N'               : len(series),
        'ADF_stat'        : round(adf_stat, 4),
        'ADF_p'           : round(adf_p, 6),
        'ADF_conclusion'  : 'Stationary' if adf_stationary else 'Non-stationary',
        'KPSS_stat'       : round(kpss_stat, 4),
        'KPSS_p'          : round(kpss_p, 4),
        'KPSS_conclusion' : 'Stationary' if kpss_stationary else 'Non-stationary',
        'Conclusion'      : conclusion
    })

# ── 3. Save results ───────────────────────────────────────────
res_df = pd.DataFrame(results)
res_df.to_csv('stationarity_results.csv', index=False)
print("\n\nResults saved: stationarity_results.csv")
print(res_df.to_string(index=False))

# ── 4. Figure: annual series + ACF ───────────────────────────
print("\nGenerating figure...")

colors = {
    'Bohicon' : '#2E75B6',
    'Cotonou' : '#1F7A4A',
    'Parakou' : '#C55A11'
}

fig = plt.figure(figsize=(16, 12))
fig.suptitle('Stationarity Analysis — Daily Rainfall in Benin (1981–2022)',
             fontsize=14, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

for idx, (name, series) in enumerate(stations.items()):
    color = colors[name]

    # Left panel: annual rainfall + trend
    ax1 = fig.add_subplot(gs[idx, 0])
    annual = series.resample('YE').sum()
    ax1.bar(annual.index.year, annual.values,
            color=color, alpha=0.7, width=0.8)
    z     = np.polyfit(annual.index.year, annual.values, 1)
    trend = np.poly1d(z)
    ax1.plot(annual.index.year, trend(annual.index.year),
             'r--', linewidth=1.8,
             label=f'Trend: {z[0]:+.1f} mm/yr')
    ax1.set_title(f'{name} — Annual rainfall', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Rainfall (mm/yr)')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Right panel: autocorrelation function (ACF)
    ax2 = fig.add_subplot(gs[idx, 1])
    plot_acf(series.values, lags=60, ax=ax2, color=color,
             title=f'{name} — Autocorrelation Function (60 lags)',
             alpha=0.05, zero=False)
    ax2.set_xlabel('Lag (days)')
    ax2.set_ylabel('ACF')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-0.15, 0.25)

plt.savefig('stationarity_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()
print("Figure saved: stationarity_analysis.png")
"""
=============================================================
STEP 3 — DFA and Hurst Exponent
=============================================================A
Input   : rainfall_benin_clean.csv
Output  : dfa_results.csv
          dfa_loglog_plots.png
          dfa_hurst_comparison.png
=============================================================

THEORETICAL BACKGROUND:
- DFA = Detrended Fluctuation Analysis (Peng et al., 1994)
- We look for the power law: F2(s) ~ s^H
- H = Hurst exponent = slope of the log-log plot
    H > 0.5  => persistent series (long-range memory)
    H < 0.5  => anti-persistent series
    H = 0.5  => uncorrelated noise (no memory)
- DFA2 = detrending with a 2nd order polynomial (recommended)
  Reference: Agbazo et al. (2019), Advances in Meteorology
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════
# DFA FUNCTION
# ════════════════════════════════════════════════════════════
def dfa(series, order=2, n_scales=40):
    """
    Computes the Detrended Fluctuation Analysis (DFA) of a time series.

    Parameters
    ----------
    series   : array-like   Time series (no NaN values)
    order    : int          Polynomial order for detrending (2 = DFA2)
    n_scales : int          Number of scales (window sizes)

    Returns
    -------
    scales : ndarray    Window sizes s
    F2     : ndarray    Fluctuation function F2(s)
    H      : float      Hurst exponent (log-log slope)
    r2     : float      R-squared of the log-log regression
    """
    x = np.array(series, dtype=float)
    N = len(x)

    # Step 1: Cumulative profile
    Y = np.cumsum(x - np.mean(x))

    # Step 2: Logarithmically spaced scales
    s_min  = max(10, order + 2)
    s_max  = N // 4
    scales = np.unique(
        np.logspace(np.log10(s_min), np.log10(s_max), n_scales).astype(int)
    )

    # Step 3: Compute F2(s) for each scale
    F2           = []
    valid_scales = []

    for s in scales:
        n_seg = N // s
        if n_seg < 4:       # need at least 4 segments
            continue

        var_list = []

        # Forward direction
        for v in range(n_seg):
            segment = Y[v*s : (v+1)*s]
            t       = np.arange(s)
            coeffs  = np.polyfit(t, segment, order)
            trend   = np.polyval(coeffs, t)
            var_list.append(np.mean((segment - trend)**2))

        # Backward direction
        for v in range(n_seg):
            start   = N - (v+1)*s
            segment = Y[start : start+s]
            t       = np.arange(s)
            coeffs  = np.polyfit(t, segment, order)
            trend   = np.polyval(coeffs, t)
            var_list.append(np.mean((segment - trend)**2))

        F2.append(np.sqrt(np.mean(var_list)))
        valid_scales.append(s)

    scales = np.array(valid_scales)
    F2     = np.array(F2)

    # Step 4: Log-log regression => H
    log_s  = np.log10(scales)
    log_F2 = np.log10(F2)
    coeffs = np.polyfit(log_s, log_F2, 1)
    H      = coeffs[0]

    # R-squared
    fitted = np.polyval(coeffs, log_s)
    ss_res = np.sum((log_F2 - fitted)**2)
    ss_tot = np.sum((log_F2 - np.mean(log_F2))**2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return scales, F2, H, r2


# ════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_csv('rainfall_benin_clean.csv', parse_dates=['date'], index_col='date')

stations = {
    'Bohicon' : df['Bohicon'].dropna(),
    'Cotonou' : df['Cotonou'].dropna(),
    'Parakou' : df['Parakou']['1981':'2012'].dropna()
}

colors = {
    'Bohicon' : '#2E75B6',
    'Cotonou' : '#1F7A4A',
    'Parakou' : '#C55A11'
}

# Season split function
def get_season(series, season):
    if season == 'rainy':
        return series[series.index.month.isin([4, 5, 6, 7, 8, 9, 10])]
    else:
        return series[series.index.month.isin([11, 12, 1, 2, 3])]


# ════════════════════════════════════════════════════════════
# RUN DFA: FULL SERIES + SEASONS
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("       DFA — HURST EXPONENT ESTIMATION")
print("="*60)

results     = []
dfa_outputs = {}

for name, series in stations.items():
    print(f"\n{'─'*50}")
    print(f"  Station: {name}   (N = {len(series)} days)")
    print(f"{'─'*50}")

    dfa_outputs[name] = {}

    for period, label in [('full',  'Full period'),
                           ('rainy', 'Rainy season'),
                           ('dry',   'Dry season')]:

        s = series if period == 'full' else get_season(series, period)

        if len(s) < 500:
            print(f"  [{label}] Series too short ({len(s)} pts), skipped.")
            continue

        scales, F2, H, r2 = dfa(s, order=2, n_scales=40)

        if   H > 0.5: interpretation = "Persistent (long-range memory)"
        elif H < 0.5: interpretation = "Anti-persistent"
        else:         interpretation = "Uncorrelated noise"

        print(f"\n  [{label}]")
        print(f"    N  = {len(s)}")
        print(f"    H  = {H:.4f}   R2 = {r2:.4f}")
        print(f"    => {interpretation}")

        dfa_outputs[name][period] = {
            'scales': scales, 'F2': F2,
            'H': H, 'r2': r2, 'N': len(s)
        }

        results.append({
            'Station'       : name,
            'Period'        : label,
            'N'             : len(s),
            'H'             : round(H, 4),
            'R2'            : round(r2, 4),
            'Interpretation': interpretation
        })


# ════════════════════════════════════════════════════════════
# SAVE RESULTS
# ════════════════════════════════════════════════════════════
res_df = pd.DataFrame(results)
res_df.to_csv('dfa_results.csv', index=False)
print("\n\nResults saved: dfa_results.csv")
print(res_df.to_string(index=False))


# ════════════════════════════════════════════════════════════
# FIGURE 1: Log-log DFA plots (full period)
# ════════════════════════════════════════════════════════════
print("\nGenerating Figure 1 (log-log plots)...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('DFA2 — Log-Log Plots (Full Period)',
             fontsize=13, fontweight='bold')

for ax, (name, outputs) in zip(axes, dfa_outputs.items()):
    if 'full' not in outputs:
        continue
    out    = outputs['full']
    scales = out['scales']
    F2     = out['F2']
    H      = out['H']
    r2     = out['r2']
    color  = colors[name]

    ax.scatter(np.log10(scales), np.log10(F2),
               color=color, s=30, alpha=0.85, zorder=3,
               label='F2(s)')

    log_s  = np.log10(scales)
    fitted = np.polyval(np.polyfit(log_s, np.log10(F2), 1), log_s)
    ax.plot(log_s, fitted, 'k--', linewidth=2,
            label=f'H = {H:.4f}  (R²={r2:.3f})')

    ax.set_title(name, fontsize=12, fontweight='bold', color=color)
    ax.set_xlabel('log₁₀(s)  [scale in days]', fontsize=10)
    ax.set_ylabel('log₁₀(F₂(s))', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax.annotate(f'H = {H:.4f}',
                xy=(0.05, 0.90), xycoords='axes fraction',
                fontsize=13, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig('dfa_loglog_plots.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: dfa_loglog_plots.png")


# ════════════════════════════════════════════════════════════
# FIGURE 2: Hurst exponent comparison (by station and season)
# ════════════════════════════════════════════════════════════
print("\nGenerating Figure 2 (Hurst comparison)...")

h_table = {}
for r in results:
    st = r['Station']
    pe = r['Period']
    if st not in h_table:
        h_table[st] = {}
    h_table[st][pe] = r['H']

station_names = list(h_table.keys())
periods       = ['Full period', 'Rainy season', 'Dry season']
period_colors = ['#444444',     '#2196F3',      '#FF9800']

x     = np.arange(len(station_names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

for i, (period, pc) in enumerate(zip(periods, period_colors)):
    h_vals = [h_table[st].get(period, np.nan) for st in station_names]
    bars   = ax.bar(x + i*width, h_vals, width,
                    label=period, color=pc,
                    alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, h_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}',
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5,
           label='H = 0.5 (uncorrelated)', alpha=0.8)

ax.set_xlabel('Station', fontsize=12)
ax.set_ylabel('Hurst Exponent H', fontsize=12)
ax.set_title('Hurst Exponent by Station and Season (DFA2)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(station_names, fontsize=11)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('dfa_hurst_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: dfa_hurst_comparison.png")
print("\nSTEP 3 COMPLETE.")
"""
=============================================================
STEP 4 — MF-DFA (Multifractal Analysis)
=============================================================
Author  : [Your Name]
Input   : rainfall_benin_clean.csv
Output  : mfdfa_results.csv + 4 figures
=============================================================
KEY FIX: We work on WET DAYS ONLY (rainfall > 0.1 mm).
Zero-rainfall days cause severe numerical instability in
MF-DFA at negative q values. This is documented in:
  - Kantelhardt et al. (2002)
  - Gires et al. (2012)
  - Agbazo et al. (2019)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MFDFA import MFDFA
import warnings
warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════
# HELPER: compute f(alpha) from h(q)
# ════════════════════════════════════════════════════════════
def compute_spectrum(q, hq):
    tau    = q * hq - 1
    alpha  = np.gradient(tau, q)
    falpha = q * alpha - tau
    mask   = (falpha >= -0.05) & (falpha <= 1.05)
    alpha  = alpha[mask]
    falpha = falpha[mask]
    if len(alpha) < 3:
        return np.array([]), np.array([]), np.nan, np.nan, np.nan
    alpha0      = alpha[np.argmax(falpha)]
    delta_alpha = alpha.max() - alpha.min()
    asymmetry   = (alpha.max() - alpha0) - (alpha0 - alpha.min())
    return alpha, falpha, alpha0, delta_alpha, asymmetry


# ════════════════════════════════════════════════════════════
# HELPER: run MF-DFA
# ════════════════════════════════════════════════════════════
def run_mfdfa(series, q_min=-3, q_max=3, n_q=31, order=2, n_scales=40):
    """
    Runs MF-DFA on WET DAYS ONLY (series > 0.1 mm).
    Uses q in [-3, +3] to avoid numerical instability.
    """
    # ── Keep only wet days ───────────────────────────────────
    x = np.array(series.dropna(), dtype=float)
    x = x[x > 0.1]          # remove zero-rainfall days
    N = len(x)

    if N < 200:
        print(f"    WARNING: only {N} wet days — results may be unreliable.")

    q = np.linspace(q_min, q_max, n_q)
    q = q[np.abs(q) > 0.1]  # avoid q very close to 0

    s_min = max(10, order + 2)
    s_max = N // 4
    if s_max <= s_min:
        return None

    lag = np.unique(
        np.logspace(np.log10(s_min), np.log10(s_max), n_scales).astype(int)
    )

    lag_out, dfa_out = MFDFA(x, lag=lag, q=q, order=order)

    hq = np.zeros(len(q))
    for i in range(len(q)):
        log_lag = np.log(lag_out)
        log_Fq  = np.log(dfa_out[:, i])
        mask    = np.isfinite(log_lag) & np.isfinite(log_Fq)
        if mask.sum() > 3:
            hq[i], _ = np.polyfit(log_lag[mask], log_Fq[mask], 1)
        else:
            hq[i] = np.nan

    valid = np.isfinite(hq)
    q_v   = q[valid]
    hq_v  = hq[valid]

    alpha, falpha, alpha0, delta_alpha, asymmetry = compute_spectrum(q_v, hq_v)

    return {
        'q'          : q_v,
        'hq'         : hq_v,
        'alpha'      : alpha,
        'falpha'     : falpha,
        'alpha0'     : alpha0,
        'delta_alpha': delta_alpha,
        'asymmetry'  : asymmetry,
        'N_wet'      : N,
        'lag'        : lag_out,
        'dfa_out'    : dfa_out
    }


def run_shuffled(series, **kwargs):
    x_wet  = series.dropna().values
    x_wet  = x_wet[x_wet > 0.1]
    x_shuf = x_wet.copy()
    np.random.shuffle(x_shuf)
    return run_mfdfa(pd.Series(x_shuf), **kwargs)


# ════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_csv('rainfall_benin_clean.csv', parse_dates=['date'], index_col='date')

stations = {
    'Bohicon' : df['Bohicon'].dropna(),
    'Cotonou' : df['Cotonou'].dropna(),
    'Parakou' : df['Parakou']['1981':'2012'].dropna()
}

colors = {'Bohicon': '#2E75B6', 'Cotonou': '#1F7A4A', 'Parakou': '#C55A11'}

def get_season(series, season):
    if season == 'rainy':
        return series[series.index.month.isin([4,5,6,7,8,9,10])]
    return series[series.index.month.isin([11,12,1,2,3])]


# ════════════════════════════════════════════════════════════
# RUN MF-DFA
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MF-DFA — WET DAYS ONLY — q in [-3, +3]")
print("="*60)

all_results  = {}
summary_rows = []

for name, series in stations.items():
    print(f"\n{'─'*55}")
    print(f"  Station: {name}   (N total = {len(series)} days)")
    wet_total = (series > 0.1).sum()
    print(f"  Wet days: {wet_total} ({100*wet_total/len(series):.1f}%)")
    print(f"{'─'*55}")
    all_results[name] = {}

    for period, label in [('full','Full period'),
                           ('rainy','Rainy season'),
                           ('dry','Dry season')]:

        s = series if period == 'full' else get_season(series, period)
        wet_days = (s > 0.1).sum()

        if wet_days < 200:
            print(f"  [{label}] Too few wet days ({wet_days}), skipped.")
            continue

        print(f"\n  [{label}]  Wet days = {wet_days}")

        res      = run_mfdfa(s)
        res_shuf = run_shuffled(s)

        if res is None:
            continue

        res['shuffled'] = res_shuf
        all_results[name][period] = res

        da_orig = res['delta_alpha']
        da_shuf = res_shuf['delta_alpha'] if res_shuf else np.nan
        ratio   = da_shuf / da_orig if (da_orig and da_orig > 0) else np.nan

        if   np.isnan(ratio): source = "Unknown"
        elif ratio > 0.8:     source = "PDF (broad distribution)"
        elif ratio < 0.3:     source = "LRC (long-range correlations)"
        else:                 source = "Mixed (LRC + PDF)"

        h_at_2 = res['hq'][np.argmin(np.abs(res['q'] - 2))]

        print(f"    h(q=2)      = {h_at_2:.4f}")
        print(f"    alpha0      = {res['alpha0']:.4f}")
        print(f"    Delta_alpha = {da_orig:.4f}  (original)")
        print(f"    Delta_alpha = {da_shuf:.4f}  (shuffled, ratio={ratio:.2f})")
        print(f"    Asymmetry   = {res['asymmetry']:.4f}")
        print(f"    Source      : {source}")

        summary_rows.append({
            'Station'          : name,
            'Period'           : label,
            'N_wet'            : wet_days,
            'h_q2'             : round(h_at_2, 4),
            'alpha0'           : round(res['alpha0'], 4),
            'delta_alpha'      : round(da_orig, 4),
            'asymmetry'        : round(res['asymmetry'], 4),
            'delta_alpha_shuf' : round(da_shuf, 4) if not np.isnan(da_shuf) else np.nan,
            'ratio'            : round(ratio, 3) if not np.isnan(ratio) else np.nan,
            'source'           : source
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('mfdfa_results.csv', index=False)
print("\n\nResults saved: mfdfa_results.csv")
print(summary_df.to_string(index=False))


# ════════════════════════════════════════════════════════════
# FIGURE 1: h(q) curves
# ════════════════════════════════════════════════════════════
print("\nGenerating Figure 1: h(q)...")
fig, ax = plt.subplots(figsize=(9, 6))

for name in stations:
    if 'full' not in all_results[name]:
        continue
    res   = all_results[name]['full']
    color = colors[name]
    ax.plot(res['q'], res['hq'], 'o-',
            color=color, linewidth=2, markersize=4, label=name)
    if 'shuffled' in res and res['shuffled']:
        shuf = res['shuffled']
        ax.plot(shuf['q'], shuf['hq'], '--',
                color=color, linewidth=1, alpha=0.5,
                label=f'{name} (shuffled)')

ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.5,
           label='h = 0.5 (monofractal)')
ax.axvline(2,   color='gray', linestyle=':', linewidth=1, alpha=0.4)
ax.set_xlabel('q  (moment order)', fontsize=12)
ax.set_ylabel('h(q)  (generalized Hurst exponent)', fontsize=12)
ax.set_title('Generalized Hurst Exponent h(q) — MF-DFA2 (wet days only)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('mfdfa_hq_curves.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: mfdfa_hq_curves.png")


# ════════════════════════════════════════════════════════════
# FIGURE 2: tau(q)
# ════════════════════════════════════════════════════════════
print("\nGenerating Figure 2: tau(q)...")
fig, ax = plt.subplots(figsize=(9, 6))

for name in stations:
    if 'full' not in all_results[name]:
        continue
    res = all_results[name]['full']
    tau = res['q'] * res['hq'] - 1
    ax.plot(res['q'], tau, 'o-',
            color=colors[name], linewidth=2, markersize=4, label=name)

ax.set_xlabel('q  (moment order)', fontsize=12)
ax.set_ylabel('tau(q)  (Renyi exponent)', fontsize=12)
ax.set_title('Renyi Exponent tau(q) — MF-DFA2 (wet days only)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('mfdfa_tauq_curves.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: mfdfa_tauq_curves.png")


# ════════════════════════════════════════════════════════════
# FIGURE 3: f(alpha) spectra
# ════════════════════════════════════════════════════════════
print("\nGenerating Figure 3: f(alpha)...")
fig, ax = plt.subplots(figsize=(9, 6))

for name in stations:
    if 'full' not in all_results[name]:
        continue
    res   = all_results[name]['full']
    color = colors[name]
    if len(res['alpha']) == 0:
        continue
    da = res['delta_alpha']
    a0 = res['alpha0']
    ax.plot(res['alpha'], res['falpha'], 'o-',
            color=color, linewidth=2, markersize=4,
            label=f"{name}  (Δα={da:.3f}, α₀={a0:.3f})")
    ax.scatter([a0], [res['falpha'].max()],
               color=color, s=100, zorder=5, marker='*')

ax.set_xlabel('alpha  (Holder exponent)', fontsize=12)
ax.set_ylabel('f(alpha)  (singularity spectrum)', fontsize=12)
ax.set_title('Multifractal Spectrum f(alpha) — MF-DFA2 (wet days only)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(-0.1, 1.2)
plt.tight_layout()
plt.savefig('mfdfa_falpha_spectra.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: mfdfa_falpha_spectra.png")


# ════════════════════════════════════════════════════════════
# FIGURE 4: South-North comparison
# ════════════════════════════════════════════════════════════
print("\nGenerating Figure 4: South-North comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('Multifractal Parameters — South to North (wet days only)',
             fontsize=13, fontweight='bold')

param_keys    = [('h_q2','Hurst exponent h(q=2)'),
                 ('delta_alpha','Spectral width Delta_alpha'),
                 ('asymmetry','Asymmetry index a_s')]
station_names = list(all_results.keys())
period_colors = {'Full period':'#333333',
                 'Rainy season':'#2196F3',
                 'Dry season':'#FF9800'}
x     = np.arange(len(station_names))
width = 0.25

for ax, (key, ylabel) in zip(axes, param_keys):
    for i, (period, pc) in enumerate(period_colors.items()):
        vals = []
        for st in station_names:
            row = summary_df[(summary_df['Station']==st) &
                             (summary_df['Period']==period)]
            vals.append(float(row[key].values[0]) if len(row)>0 else np.nan)

        bars = ax.bar(x + i*width, vals, width, label=period,
                      color=pc, alpha=0.85,
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f'{val:.3f}',
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

    if key == 'h_q2':
        ax.axhline(0.5, color='red', linestyle='--',
                   linewidth=1.2, alpha=0.7, label='H=0.5 ref.')

    ax.set_title(ylabel, fontsize=11, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(station_names, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('mfdfa_southnorth.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: mfdfa_southnorth.png")
print("\nSTEP 4 COMPLETE.")
"""
=============================================================
STEP 5 — Final Figures for the AIMS Essay
=============================================================
Author  : [Your Name]
Input   : rainfall_benin_clean.csv
          dfa_results.csv / mfdfa_results.csv
Output  : fig_seasonal_spectra.png
          fig_summary_table.png
=============================================================
Run AFTER step03 and step04.
KEY FIX: wet days only + q in [-3, +3]
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MFDFA import MFDFA
import warnings
warnings.filterwarnings('ignore')


def compute_spectrum(q, hq):
    tau    = q * hq - 1
    alpha  = np.gradient(tau, q)
    falpha = q * alpha - tau
    mask   = (falpha >= -0.05) & (falpha <= 1.05)
    alpha  = alpha[mask]
    falpha = falpha[mask]
    if len(alpha) < 3:
        return np.array([]), np.array([]), np.nan, np.nan
    delta_alpha = alpha.max() - alpha.min()
    alpha0      = alpha[np.argmax(falpha)]
    return alpha, falpha, delta_alpha, alpha0


def run_mfdfa_simple(series, order=2, n_scales=35):
    # Wet days only
    x = np.array(series.dropna(), dtype=float)
    x = x[x > 0.1]
    N = len(x)
    if N < 200:
        return np.array([]), np.array([]), np.nan, np.nan

    q   = np.linspace(-3, 3, 31)
    q   = q[np.abs(q) > 0.1]
    lag = np.unique(
        np.logspace(np.log10(10), np.log10(N // 4), n_scales).astype(int)
    )
    if len(lag) < 5:
        return np.array([]), np.array([]), np.nan, np.nan

    lag_out, dfa_out = MFDFA(x, lag=lag, q=q, order=order)
    hq = np.array([
        np.polyfit(np.log(lag_out), np.log(dfa_out[:, i]), 1)[0]
        if np.all(np.isfinite(np.log(np.abs(dfa_out[:, i]) + 1e-30)))
        else np.nan
        for i in range(len(q))
    ])
    valid = np.isfinite(hq)
    if valid.sum() < 5:
        return np.array([]), np.array([]), np.nan, np.nan

    alpha, falpha, da, a0 = compute_spectrum(q[valid], hq[valid])
    return alpha, falpha, da, a0


# ── Load data ─────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('rainfall_benin_clean.csv', parse_dates=['date'], index_col='date')

stations = {
    'Bohicon' : df['Bohicon'].dropna(),
    'Cotonou' : df['Cotonou'].dropna(),
    'Parakou' : df['Parakou']['1981':'2012'].dropna()
}
colors = {'Bohicon':'#2E75B6', 'Cotonou':'#1F7A4A', 'Parakou':'#C55A11'}

def get_season(series, season):
    if season == 'rainy':
        return series[series.index.month.isin([4,5,6,7,8,9,10])]
    return series[series.index.month.isin([11,12,1,2,3])]


# ════════════════════════════════════════════════════════════
# FIGURE: Seasonal f(alpha) — Figure 4.6
# ════════════════════════════════════════════════════════════
print("\nGenerating: fig_seasonal_spectra.png ...")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('Seasonal Comparison of Multifractal Spectra f(alpha) — wet days only',
             fontsize=13, fontweight='bold')

for ax, (name, series) in zip(axes, stations.items()):
    color = colors[name]

    print(f"  {name} — Full period ...")
    a_full, f_full, da_full, a0_full = run_mfdfa_simple(series)
    if len(a_full) > 0:
        ax.plot(a_full, f_full, 'o-', color=color, linewidth=2.5,
                markersize=4, label=f'Full  (Δα={da_full:.3f})')

    s_rain = get_season(series, 'rainy')
    print(f"  {name} — Rainy season (N wet={(s_rain>0.1).sum()}) ...")
    a_rain, f_rain, da_rain, _ = run_mfdfa_simple(s_rain)
    if len(a_rain) > 0:
        ax.plot(a_rain, f_rain, 's--', color='#2196F3', linewidth=2,
                markersize=4, label=f'Rainy (Δα={da_rain:.3f})')

    s_dry = get_season(series, 'dry')
    print(f"  {name} — Dry season (N wet={(s_dry>0.1).sum()}) ...")
    a_dry, f_dry, da_dry, _ = run_mfdfa_simple(s_dry)
    if len(a_dry) > 0:
        ax.plot(a_dry, f_dry, '^:', color='#FF9800', linewidth=2,
                markersize=4, label=f'Dry   (Δα={da_dry:.3f})')

    ax.set_title(name, fontsize=12, fontweight='bold', color=color)
    ax.set_xlabel('alpha  (Holder exponent)', fontsize=11)
    ax.set_ylabel('f(alpha)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, 1.2)

plt.tight_layout()
plt.savefig('fig_seasonal_spectra.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: fig_seasonal_spectra.png")


# ════════════════════════════════════════════════════════════
# FIGURE: Summary table
# ════════════════════════════════════════════════════════════
print("\nGenerating: fig_summary_table.png ...")

dfa_res   = pd.read_csv('dfa_results.csv')
mfdfa_res = pd.read_csv('mfdfa_results.csv')

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

cols = ['Station', 'Period', 'N wet days',
        'H (DFA2)', 'alpha0', 'Delta_alpha', 'Asymmetry', 'Source']
rows = []

for _, row in mfdfa_res.iterrows():
    st    = row['Station']
    pe    = row['Period']
    h_row = dfa_res[(dfa_res['Station']==st) & (dfa_res['Period']==pe)]
    H_val = f"{h_row['H'].values[0]:.4f}" if len(h_row) > 0 else '—'
    rows.append([
        st, pe, int(row['N_wet']),
        H_val,
        f"{row['alpha0']:.4f}",
        f"{row['delta_alpha']:.4f}",
        f"{row['asymmetry']:.4f}",
        row['source']
    ])

table = ax.table(cellText=rows, colLabels=cols,
                 cellLoc='center', loc='center', bbox=[0,0,1,1])
table.auto_set_font_size(False)
table.set_fontsize(9)

for j in range(len(cols)):
    table[0, j].set_facecolor('#2E75B6')
    table[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows)+1):
    for j in range(len(cols)):
        if i % 2 == 0:
            table[i, j].set_facecolor('#F0F4FA')

ax.set_title('Summary of Multifractal Parameters — Benin Daily Rainfall (wet days only)',
             fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('fig_summary_table.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: fig_summary_table.png")
print("\nSTEP 5 COMPLETE.")
