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
* vectorized sky searches
* precomputation and caching of the **M matrix**

Typical usage:
```python
fstat = GPU_FeStat(psrs)
fstatz = fstat.compute_Fe(f0, skyloc)
```

The result `fstatz` contains the Fe-statistic evaluated at each sky location.

---

# Run Scripts

### `run_GPU_Fe_statistics.py` — single injection
Minimal working example for a **single CW injection**. The script:
1. Loads a set of pulsars
2. Injects a continuous GW signal with fixed parameters
3. Computes the Fe-statistic over a HEALPix sky grid
4. Produces a Mollweide sky map of the statistic

### `run_multiple_GPU_Fe_statistics.py` — batch of injections
Extended script for running the Fe-statistic over **multiple injections in a single vmapped pass**. The script:
1. Loads a set of pulsars
2. Generates N injections with randomised sky position and orientation via `generate_residual_samples_vmap`
3. Computes Fe-statistic maps for all injections at once via `GPU_FeStat.compute_Fe_batch`
4. Returns an array of shape `(N, n_sky)` — one Fe map per injection

Typical usage:
```python
fstat = GPU_FeStat(psrs)
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