import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import healpy as hp
import discovery as ds
from injection_discovery import run_injection, makemodel_cgw
from GPU_Fe_statistics import GPU_FeStat


def generate_residual_samples_vmap(
    d_psrs,
    n_samples:  int   = 10_000,
    chunk_size: int   = 512,
    base_seed:  int   = 42,
    log10_f0:   float = -8.0,
    log10_h0:   float = -11.5,
    log10_Mc:   float = 9.2,
    background: bool  = False,
    bg_params:  dict  = None,
):
    """
    Generate N simulated PTA residuals by vmapping the CW injection sampler.

    Sky position and orientation are drawn uniformly at random for each sample;
    frequency, amplitude, and chirp mass are kept fixed.

    Parameters
    ----------
    background : bool
        If True the model includes a correlated-red-noise global GP and the
        background parameters in bg_params are injected in every sample.
    bg_params  : dict or None
        e.g. {'crn_log10_A': -15.0, 'crn_gamma': 4.33}.
        Required when background=True.

    Returns
    -------
    all_residuals : list of chunks; each chunk is a pytree with one array
                    per pulsar of shape (chunk_N, n_toa_i).
    all_params    : list of param dicts per chunk.
    model         : the GlobalLikelihood (returned so the caller can pass it
                    to GPU_FeStat for the recovery step).
    """
    # CW common parameter names — never include background params here
    cwcommon = [
        'cw_sindec', 'cw_cosinc', 'cw_log10_f0', 'cw_log10_h0',
        'cw_phi_earth', 'cw_psi', 'cw_ra', 'cw_log10_Mc',
    ]

    model   = makemodel_cgw(d_psrs, cwcommon, background=background)
    sampler = model.sample

    # CW parameters randomised across the batch
    batched_keys = {'cw_sindec', 'cw_cosinc', 'cw_phi_earth', 'cw_psi', 'cw_ra'}

    # CW parameters fixed for the whole run
    fixed_cw = {
        'cw_log10_f0': log10_f0,
        'cw_log10_h0': log10_h0,
        'cw_log10_Mc': log10_Mc,
    }

    # Dummy pulsar distance (1 kpc) — only the Earth term is injected.
    # phi_psr must be present per-pulsar (discovery maps it by name); 0 = coherent.
    for p in d_psrs:
        fixed_cw[f'{p.name}_cw_d_psr']   = 1.0
        fixed_cw[f'{p.name}_cw_phi_psr'] = 0.0

    # Noise parameters
    noise_params = {}
    for p in d_psrs:
        noise_params.update(p.noisedict)

    # Background parameters (fixed scalars, not batched)
    if background:
        if bg_params is None:
            bg_params = {'crn_log10_A': -15.0, 'crn_gamma': 4.33}
        fixed_cw.update(bg_params)

    rng = jax.random.key(base_seed)
    all_residuals, all_params = [], []
    n_chunks = (n_samples + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        N = min(chunk_size, n_samples - i * chunk_size)

        rng, k_sindec, k_cosinc, k_phi, k_psi, k_ra, k_sample = jax.random.split(rng, 7)

        batched_cw = {
            'cw_sindec':    jax.random.uniform(k_sindec, (N,), minval=-1.0,      maxval=1.0),
            'cw_cosinc':    jax.random.uniform(k_cosinc, (N,), minval=-1.0,      maxval=1.0),
            'cw_phi_earth': jax.random.uniform(k_phi,    (N,), minval=0.0,       maxval=2*jnp.pi),
            'cw_psi':       jax.random.uniform(k_psi,    (N,), minval=0.0,       maxval=jnp.pi),
            'cw_ra':        jax.random.uniform(k_ra,     (N,), minval=0.0,       maxval=2*jnp.pi),
        }

        full_params = {**noise_params, **fixed_cw, **batched_cw}

        in_axes_params = {k: (0 if k in batched_keys else None) for k in full_params}

        sampler_vmap = jax.vmap(sampler, in_axes=(0, in_axes_params))

        keys = jax.random.split(k_sample, N)
        _, residuals_chunk = sampler_vmap(keys, full_params)

        all_residuals.append(jax.device_get(residuals_chunk))
        all_params.append({
            k: np.array(v) if k in batched_keys else v
            for k, v in full_params.items()
        })

        print(f"  chunk {i+1}/{n_chunks} ({min((i+1)*chunk_size, n_samples)}/{n_samples})")

    return all_residuals, all_params, model


# ══════════════════════════════════════════════════════════════════════════════
#  TOGGLE
# ══════════════════════════════════════════════════════════════════════════════
background = False
bg_params  = {'crn_log10_A': -14.8, 'crn_gamma': 13/3}

# ── Load pulsars ──────────────────────────────────────────────────────────────

path = './EPTA_dr2/gwb20nHz_-14._feathers/'
s_dsfiles = np.sort(os.listdir(path))
d_psrs = [ds.Pulsar.read_feather(path + f'{psrfile}') for psrfile in s_dsfiles]


# ── Generate N injections ─────────────────────────────────────────────────────

residuals, params, inj_model = generate_residual_samples_vmap(
    d_psrs,
    n_samples  = 100,
    chunk_size = 25,
    base_seed  = 42,
    log10_f0   = -8.3,
    log10_h0   = -15,
    log10_Mc   = 9.2,
    background = background,
    bg_params  = bg_params if background else None,
)

print(f"Generated {sum(c[0].shape[0] for c in residuals)} samples "
      f"in {len(residuals)} chunks.")


# ── Sky grid ──────────────────────────────────────────────────────────────────

nside   = 16
nskyloc = hp.nside2npix(nside)
skyloc  = np.array(hp.pix2ang(nside, np.arange(nskyloc)))


# ── Fe-statistic batch computation ───────────────────────────────────────────

f0 = 10**(-8.3)

if background:
    fstat = GPU_FeStat(d_psrs, lik=inj_model, params=bg_params, n_crn=60)
else:
    fstat = GPU_FeStat(d_psrs)

start = time.time()
all_fe_maps = jnp.concatenate(
    [fstat.compute_Fe_batch(f0, skyloc, chunk) for chunk in residuals],
    axis=0,
)
print(f"Fe maps computed in {time.time() - start:.1f} s — shape: {all_fe_maps.shape}")
