import os
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import healpy as hp
import discovery as ds

from injection_discovery import run_injection
from GPU_Fe_statistics import GPU_FeStat

# --- Load the pulsars ---
path = './EPTA_dr2/gwb20nHz_-14._feathers/'
s_dsfiles = np.sort(os.listdir(path))
d_psrs = [ds.Pulsar.read_feather(path + f'{psrfile}') for psrfile in s_dsfiles]

# --- Injection parameters ---
cwcommon = ['cw_sindec', 'cw_cosinc', 'cw_log10_f0', 'cw_log10_h0',
            'cw_phi_earth', 'cw_psi', 'cw_ra', 'cw_log10_Mc']

log10_h0  = -11.5
log10_f0  = -8.3
log10_M0  =  9.2
ra        =  1.35
sindec    = -0.2
cosinc    =  0.2
psi       =  1.2
phi_earth =  1.5

cwpars = [sindec, cosinc, log10_f0, log10_h0, phi_earth, psi, ra, log10_M0]

# --- Injection ---
residuals, inj_model = run_injection(d_psrs, cwpars, cwcommon, pulsterm=False)
for ii, psr in enumerate(d_psrs):
    psr.residuals = np.array(residuals[ii]).squeeze()

# --- Sky grid ---
nside   = 16
nskyloc = hp.nside2npix(nside)
skyloc  = np.array(hp.pix2ang(nside, np.arange(nskyloc)))

# --- Fe-statistic ---
f0 = 10**log10_f0

fstat = GPU_FeStat(d_psrs)
fstat.precompute_M(f0)

start  = time.time()
fstatz = fstat.compute_Fe(f0, skyloc)
print('Time taken by the Fe computation:', time.time() - start, 's')

# --- Plot with CGW marker ---
# Convert the CGW position to healpy coordinates (theta, phi)
# healpy uses theta = colatitude [0, pi], phi = longitude [0, 2pi]
dec       = np.arcsin(sindec)        # declination in radians
cgw_theta = np.pi / 2 - dec         # colatitude
cgw_phi   = ra                       # longitude = right ascension

fig = plt.figure(figsize=(10, 6))
hp.mollview(np.array(fstatz), rot=180, title='Fe-statistic', hold=True)

# Project the CGW position onto the Mollweide map
# hp.mollview with rot=180 requires rotating phi by 180 degrees for the marker
hp.projscatter(cgw_theta, cgw_phi,
               marker='x',
               color='green',
               s=100,
               linewidths=1.5,
               label='Injected CGW',
               zorder=10)

plt.legend(loc='lower right', framealpha=0.7)
outfile = 'festat_map.png'
plt.savefig(outfile, dpi=150, bbox_inches='tight')
plt.show()
print(f'Figure saved in: {outfile}')