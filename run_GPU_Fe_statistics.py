import os
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import healpy as hp
import discovery as ds

from injection_discovery import run_injection
from GPU_Fe_statistics import GPU_FeStat

# ── Load the pulsars ──────────────────────────────────────────────────────────
path = '/home/ludovicarbone/sequential_sbi_maps/feathers/25_EPTA_positions_feathers/'
s_dsfiles = np.sort(os.listdir(path))
d_psrs = [ds.Pulsar.read_feather(path + f'{psrfile}') for psrfile in s_dsfiles]

# ══════════════════════════════════════════════════════════════════════════════
#  TOGGLE: set to True to inject AND recover with a CRN background
# ══════════════════════════════════════════════════════════════════════════════
background = True

# Background parameters (only used when background=True)
bg_params = {'crn_log10_A': -14.8, 'crn_gamma': 13/3}

# ── CW common parameter names (no background params here) ────────────────────
cwcommon = ['cw_sindec', 'cw_cosinc', 'cw_log10_f0', 'cw_log10_h0',
            'cw_phi_earth', 'cw_psi', 'cw_ra', 'cw_log10_Mc']

# ── CW injection parameters ──────────────────────────────────────────────────
log10_h0  = -13.5
log10_f0  = -8.3
log10_M0  =  9.2
ra        =  1.35
sindec    = -0.2
cosinc    =  0.2
psi       =  1.2
phi_earth =  1.5

cwpars = [sindec, cosinc, log10_f0, log10_h0, phi_earth, psi, ra, log10_M0]

# ── Injection ─────────────────────────────────────────────────────────────────
residuals, inj_model = run_injection(
    d_psrs, cwpars, cwcommon,
    pulsterm=False,
    background=background,
    bg_params=bg_params if background else None,
)
for ii, psr in enumerate(d_psrs):
    psr.residuals = np.array(residuals[ii]).squeeze()

# ── Sky grid ──────────────────────────────────────────────────────────────────
nside   = 16
nskyloc = hp.nside2npix(nside)
skyloc  = np.array(hp.pix2ang(nside, np.arange(nskyloc)))

# ── Fe-statistic ──────────────────────────────────────────────────────────────
f0 = 10**log10_f0

if background:
    # Pass the likelihood + background params → CRN marginalised
    fstat = GPU_FeStat(d_psrs, lik=inj_model, params=bg_params, n_crn=60)
else:
    # White noise + timing model only
    fstat = GPU_FeStat(d_psrs)

fstat.precompute_M(f0)

start  = time.time()
fstatz = fstat.compute_Fe(f0, skyloc)
print('Time taken by the Fe computation:', time.time() - start, 's')

# ── Plot with CGW marker ─────────────────────────────────────────────────────
dec       = np.arcsin(sindec)
cgw_theta = np.pi / 2 - dec
cgw_phi   = ra

fig = plt.figure(figsize=(10, 6))
title = 'Fe-statistic (with CRN background)' if background else 'Fe-statistic (white noise only)'
hp.mollview(np.array(fstatz), rot=180, title=title, hold=True)

hp.projscatter(cgw_theta, cgw_phi,
               marker='x', color='green', s=100, linewidths=1.5,
               label='Injected CGW', zorder=10)

plt.legend(loc='lower right', framealpha=0.7)
outfile = 'festat_map.png'
plt.savefig(outfile, dpi=150, bbox_inches='tight')
plt.show()
print(f'Figure saved in: {outfile}')
