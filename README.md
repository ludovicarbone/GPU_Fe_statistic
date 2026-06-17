# GPU Fe-statistic for Continuous Gravitational Wave Searches in PTAs
This repository contains a **GPU-accelerated implementation of the Fe-statistic** for searches of **continuous gravitational waves (CGWs)** in Pulsar Timing Array (PTA) data.
The code uses **JAX** to perform fast computations and evaluate the Fe-statistic across a grid of sky locations.
The implementation is designed to work with pulsar datasets compatible with the **Discovery framework**.

---

# Repository Structure
```
.
├── GPU_Fe_statistics.py
├── injection_discovery.py
├── run_GPU_Fe_statistics.py
├── run_multiple_GPU_Fe_statistics.py
└── README.md
```

### `GPU_Fe_statistics.py`
Main implementation of the **Fe-statistic**.
This module provides the `GPU_FeStat` class, which computes the Fe-statistic for a PTA dataset at a given gravitational wave frequency and over a grid of sky positions.

Key features:
* GPU acceleration via **JAX**
* Vectorized sky searches
* Precomputation and caching of the **M matrix**
* Support for **correlated red noise (CRN) background marginalisation**: when a `GlobalLikelihood` with a global GP is provided, the inner products are computed using per-pulsar Sigma matrices that include both the timing model and the CRN basis. When no background is provided, the code falls back to white noise + timing model only.

Typical usage:
```python
# --- White noise only ---
fstat = GPU_FeStat(psrs)
fstat.precompute_M(f0)
fstatz = fstat.compute_Fe(f0, skyloc)

# --- With CRN background marginalisation ---
fstat = GPU_FeStat(psrs, lik=lik, params={'crn_log10_A': -15.0, 'crn_gamma': 4.33}, n_crn=60)
fstat.precompute_M(f0)
fstatz = fstat.compute_Fe(f0, skyloc)
```

The result `fstatz` contains the Fe-statistic evaluated at each sky location.

> **Note on the white-noise mode.** `GPU_FeStat(psrs)` with no `lik` falls back
> to raw TOA errors (`toaerrs`), which is correct only for idealised simulations
> where EFAC/EQUAD are negligible. For real data, build a white-noise likelihood
> and pass it so the Fe-statistic uses the same EFAC/EQUAD-corrected noise as the
> injection:
> ```python
> inj_model_wn = makemodel_cgw(psrs, cwcommon, background=False)
> fstat = GPU_FeStat(psrs, lik=inj_model_wn)
> ```
> A reference epoch `tref` (default: mean of all TOAs) is shared between
> injection, SNR calibration and recovery; pass `tref=...` explicitly to all
> three if you override the default.


---

# Background Marginalisation

The Fe-statistic inner products can optionally marginalise over a **correlated red noise (CRN) background** (e.g. a gravitational wave background with Hellings-Downs correlations).

This is controlled by a single `background` flag that affects both the **injection** and the **recovery** steps:

| `background` | Injection | Fe-statistic recovery |
|---|---|---|
| `False` | CW signal + white noise only | White noise + timing model |
| `True`  | CW signal + CRN + white noise | CRN marginalised via per-pulsar Sigma matrices |

When `background=True`:
* The injection model (`makemodel_cgw`) includes a global GP built with `makegp_fourier_global` and a specified overlap reduction function (default: Hellings-Downs).
* The Fe-statistic (`GPU_FeStat`) receives the `GlobalLikelihood` and background parameters, and builds per-pulsar Sigma matrices that incorporate both the timing model basis and the CRN Fourier basis (block-diagonal approximation of the full cross-pulsar Phi_inv).

---

# Run Scripts

### `run_GPU_Fe_statistics.py` — single injection
Minimal working example for a **single CW injection**. The script:
1. Loads a set of pulsars
2. Injects a continuous GW signal with fixed parameters (optionally on top of a CRN background)
3. Computes the Fe-statistic over a HEALPix sky grid
4. Produces a Mollweide sky map of the statistic

The `background` flag at the top of the script controls both injection and recovery.

### `run_multiple_GPU_Fe_statistics.py` — batch of injections
Extended script for running the Fe-statistic over **multiple injections in a single vmapped pass**.
You can produce 100k maps with a chunk size of 2000 maps in 3 minutes and 30 seconds on a single GPU.
The script:
1. Loads a set of pulsars
2. Generates N injections with randomised sky position and orientation via `generate_residual_samples_vmap`
3. Computes Fe-statistic maps for all injections at once via `GPU_FeStat.compute_Fe_batch`
4. Returns an array of shape `(N, n_sky)` — one Fe map per injection

The `background` flag at the top of the script controls both injection and recovery.

`generate_residual_samples_vmap` also returns the `model` (`GlobalLikelihood`) so that the same object can be passed directly to `GPU_FeStat` for the recovery step.

Typical usage:
```python
# --- White noise only ---
fstat = GPU_FeStat(psrs)
all_fe_maps = jnp.concatenate(
    [fstat.compute_Fe_batch(f0, skyloc, chunk) for chunk in residuals],
    axis=0,
)   # shape (N, n_sky)

# --- With CRN background ---
fstat = GPU_FeStat(psrs, lik=inj_model, params=bg_params, n_crn=60)
all_fe_maps = jnp.concatenate(
    [fstat.compute_Fe_batch(f0, skyloc, chunk) for chunk in residuals],
    axis=0,
)   # shape (N, n_sky)
```

---

# Signal Injection
Example injection utilities are provided in:
```
injection_discovery.py
```
In this repository, injections are implemented using the **Discovery framework**:
[https://github.com/nanograv/discovery](https://github.com/nanograv/discovery)

Discovery allows the injection step to remain **GPU-based and compatible with the rest of the pipeline**.
However, **the Fe-statistic implementation itself is independent of the injection method**.
Any injection method can be used as long as the pulsar residuals are updated before running the Fe-statistic search.
The file `injection_discovery.py` simply provides an **example of how to perform injections using Discovery**.

The key functions are:
* `makemodel_cgw(psrs, cwcommon, background=False, orf=ds.hd_orf)` — builds the `GlobalLikelihood`, optionally including a CRN global GP.
* `run_injection(d_psrs, cwpars, cwcommon, background=False, bg_params=None)` — performs a single injection and returns the simulated residuals and the model.

When `background=True`, the background parameters (e.g. `crn_log10_A`, `crn_gamma`) are passed separately via `bg_params` and are **not** included in `cwcommon`.

---

# Dependencies
Required Python packages:
```
numpy
jax
jaxlib
healpy
matplotlib
discovery
```
Discovery can be installed from:
[https://github.com/nanograv/discovery](https://github.com/nanograv/discovery)
