import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

FS = 18  # global font size

plt.rcParams.update({
    'font.size':        FS,
    'axes.labelsize':   FS,
    'axes.titlesize':   FS,
    'xtick.labelsize':  FS,
    'ytick.labelsize':  FS,
    'legend.fontsize':  FS,
})

# Data
J    = [0,      400,    800,    900,    1000,   1100,   1200,   1600]
qsh  = [96203,  112133, 128753, 133034, 137362, 141728, 146110, 163774]
qsc  = [96196,  108645, 122879, 126742, 130734, 134861, 139127, 157674]
eta  = [0.00681, 3.11,   4.56,   4.73,   4.83,   4.84,   4.78,   3.72]

J   = np.array(J)
qsh = np.array(qsh)
qsc = np.array(qsc)
eta = np.array(eta)

fig, ax1 = plt.subplots(figsize=(10, 6))

# ── Heat flux (left y-axis) ──────────────────────────────────────────────────
color_hot  = '#E24B4A'
color_cold = '#378ADD'

ax1.scatter(J, qsh, marker='^', color=color_hot,  s=70, zorder=4, label='Hot side')
ax1.scatter(J, qsc, marker='v', color=color_cold, s=70, zorder=4, label='Cold side')

# Vertical connector lines between ^ and v at each J
for j, h, c in zip(J, qsh, qsc):
    ax1.plot([j, j], [h, c], color='gray', linewidth=0.8,
             linestyle='--', alpha=0.5, zorder=2)

ax1.set_xlabel('Current density J (A/m²)')
ax1.set_ylabel(r'Heat flux q (W/m²)  $[\times10^3]$')
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}'))
ax1.tick_params(axis='both', labelsize=FS)  # explicit override for ticks

# ── Efficiency η (right y-axis) ──────────────────────────────────────────────
color_eta = '#1D9E75'

ax2 = ax1.twinx()
ax2.plot(J, eta, color=color_eta, linewidth=1.8,
         linestyle='--', zorder=3, label='η efficiency (%)')
ax2.scatter(J, eta, color=color_eta, s=50, zorder=4)

ax2.set_ylabel('Efficiency η (%)', color=color_eta)
ax2.tick_params(axis='y', labelcolor=color_eta, labelsize=FS)  # explicit override
ax2.set_ylim(0, 6)

# ── Legend ───────────────────────────────────────────────────────────────────
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='upper left', framealpha=0.85)

# ── Styling ──────────────────────────────────────────────────────────────────
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_xlim(-50, 1680)
fig.tight_layout()

plt.savefig('thermoelectric_plot.png', dpi=150, bbox_inches='tight')
plt.show()
