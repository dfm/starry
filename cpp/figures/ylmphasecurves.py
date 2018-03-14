"""Plot some Ylm thermal phase curves."""
import matplotlib.pyplot as pl
import numpy as np
from starry import Map

# Compute and plot up to this order
lmax = 6

# Number of points in the phase curve
nt = 100

# Set up the plot
fig, ax = pl.subplots(lmax + 1, lmax + 1, figsize=(9, 5.5))
for axis in ax.flatten():
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)
for l in range(lmax + 1):
    ax[l, 0].set_ylabel(r"$l = %d$" % l,
                        rotation='horizontal',
                        labelpad=30, y=0.38,
                        fontsize=12)
for j, m in enumerate(range(lmax + 1)):
    ax[-1, j].set_xlabel(r"$m = %d$" % m, labelpad=30, fontsize=12)

# Rotate about this vector
ux = np.array([1., 0., 0.])
uy = np.array([0., 1., 0.])
y = Map(lmax)
theta = np.linspace(0, 2 * np.pi, nt, endpoint=False)
for u, zorder in zip([ux, uy], [1, 0]):
    for i, l in enumerate(range(lmax + 1)):
        for j, m in enumerate(range(l + 1)):
            y.reset()
            y.set_coeff(l, m, 1)
            flux = y.flux(u=u, theta=theta, x0=-9, y0=-9, r=1)
            if np.max(np.abs(flux)) < 1e-10:
                flux *= 0
            ax[i, j].plot(flux, lw=1, zorder=zorder)

# Hack a legend
axleg = pl.axes([0.7, 0.7, 0.15, 0.15])
axleg.plot([0, 0], [1, 1], label=r'$\vec{u} = \hat{x}$')
axleg.plot([0, 0], [1, 1], label=r'$\vec{u} = \hat{y}$')
axleg.axis('off')
leg = axleg.legend(title=r'Phase curves', fontsize=18)
leg.get_title().set_fontsize('20')
leg.get_frame().set_linewidth(0.0)

# Save!
fig.savefig("ylmphasecurves.pdf", bbox_inches='tight')
pl.close(fig)
