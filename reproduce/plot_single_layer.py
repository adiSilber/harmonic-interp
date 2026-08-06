#!/usr/bin/env python3
"""Single-layer (layer 12) vs full-range steering — resolution distributions.

Reads the two resolutions.csv files (Step 13 and Step 18) and plots, per target key,
the unsteered baseline / full layer range / layer 12 alone. Same six categories,
same viridis stops and print geometry as the notebook's Figure 5, so the two figures
read as one system.

Writes reproduce_output/figures/single_layer_l12_vs_all.{pdf,png} and prints the
per-category counts quoted in the paper text.

CPU-only. Run from the repo root, after reproduce/run_single_layer_steering.sh.

Usage:
    python reproduce/plot_single_layer.py
"""

import csv
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── roman-numeral normalisation + the paper's six categories (from Figure 5) ──────
_PFX = re.compile(r'^([b#♯♭]*[IViv]+)([°oø+]?)(.*)$')
_MG = {'I+': 'I+/bVI+', 'I+7': 'I+/bVI+', 'bVI+': 'I+/bVI+'}
_RO = {'I','ii','iii','IV','V','vi','vii°','vii','I7','ii7','iii7','IV7','V7','vi7',
       'viiø7','vii°7','i','ii°','III','iv','v','VI','VII','i7','III7','IIIø7','iv7',
       'VI7','VII7','bII','bIII','bV','bVI','bVII','bvi','I+/bVI+','bII+','i°','vi°',
       'III°','#i°','#i°7','#ii°','#iv','II','other','none'}
_ST_DIATONIC = {'ii','iii','IV','V','vii°','I7','ii7','iii7','IV7','V7','vi7','viiø7'}
_ST_PAR = {'ii°','iv','v','bIII','bVI','bVII','iv7','i7','i+','v+'}


def _nrm(r):
    if not isinstance(r, str) or not r:
        return None
    m = _PFX.match(r)
    if not m:
        return r
    a, q, rest = m.groups()
    if q == 'o':
        q = '°'
    b = a + q
    s = bool(re.match(r'^(7|65|43|42)(?!\d)', rest))
    return (b + '7') if q == 'ø' else (b + ('7' if s else ''))


def _cn(r):
    if not r:
        return 'none'
    n = _MG.get(_nrm(r), _nrm(r))
    return n if n in _RO else 'other'


def _stack(counter):
    total = sum(counter.values())
    cats = {'I': counter.get('I', 0), 'i': counter.get('i', 0), 'vi': counter.get('vi', 0),
            'orig_rel_diaton': sum(counter.get(r, 0) for r in _ST_DIATONIC),
            'par_diatonic': sum(counter.get(r, 0) for r in _ST_PAR)}
    cats['other'] = max(0, total - sum(cats.values()))
    return cats, total


# ── paper geometry (notebook setup cell) ─────────────────────────────────────────
TEXT_W, BASE_FS = 6.75, 6.5
plt.rcParams.update({
    'figure.dpi': 200, 'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': BASE_FS, 'axes.titlesize': BASE_FS + 1, 'axes.labelsize': BASE_FS + 1,
    'xtick.labelsize': BASE_FS, 'ytick.labelsize': BASE_FS, 'legend.fontsize': BASE_FS,
    'legend.frameon': False, 'legend.handlelength': 1.1, 'legend.handletextpad': 0.5,
    'legend.labelspacing': 0.3, 'legend.borderaxespad': 0.2,
    'axes.linewidth': 0.6, 'grid.linewidth': 0.5,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
    'xtick.major.pad': 2.0, 'ytick.major.pad': 2.0,
})

FULL = 'reproduce_output/steering/resolutions.csv'
L12 = 'reproduce_output/steering_l12/resolutions.csv'
IKEYS = ['I', 'i', 'vi', 'orig_rel_diaton', 'par_diatonic', 'other']
LABELS = ['I', 'i', 'vi', 'orig. &\nrel. diatonic', 'par.\ndiatonic', 'other']
# Figure 5's viridis stops: dark -> light, so identity survives greyscale and CVD.
COLORS = [plt.cm.viridis(v) for v in (0.1, 0.45, 0.8)]

rows = {p: list(csv.DictReader(open(p))) for p in (FULL, L12)}


def series(path, cond):
    return _stack(Counter(_cn(r['rn']) for r in rows[path] if r['condition'] == cond))


PANELS = [
    ('parallel-minor steering', [
        ('unsteered', *series(FULL, 'baseline')),
        ('layers 11–15, α=0.10', *series(FULL, 'parallel')),
        ('layer 12 only, α=0.40', *series(L12, 'parallel')),
    ]),
    ('relative-minor steering', [
        ('unsteered', *series(FULL, 'baseline')),
        ('layers 11–15, α=0.15', *series(FULL, 'relative')),
        ('layer 12 only, α=0.40', *series(L12, 'relative')),
    ]),
]

x, w = np.arange(len(IKEYS)), 0.26
fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, TEXT_W / 3.1), sharey=True)
for ax, (title, ser) in zip(axes, PANELS):
    for k, (label, s, total) in enumerate(ser):
        pcts = [100 * s[key] / total if total else 0 for key in IKEYS]
        bars = ax.bar(x + (k - 1) * w, pcts, w * 0.92, label=f'{label} (n={total})',
                      color=COLORS[k])
        for bar, key in zip(bars, IKEYS):
            if s[key] > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                        str(s[key]), ha='center', va='bottom', fontsize=BASE_FS - 1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(loc='upper center')
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('% of samples')
fig.tight_layout(pad=0.3)
for ext in ('pdf', 'png'):
    out = f'reproduce_output/figures/single_layer_l12_vs_all.{ext}'
    fig.savefig(out, bbox_inches='tight', pad_inches=0.01)
    print('saved', out)

# ── the numbers for the paper text ───────────────────────────────────────────────
print()
for title, ser in PANELS:
    print(title)
    for label, s, total in ser:
        cells = '  '.join(f'{k}={s[k]:3d} ({100*s[k]/total:4.1f}%)' for k in IKEYS)
        print(f'  {label:24s} n={total:4d}  {cells}')
