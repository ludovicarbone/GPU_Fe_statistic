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
├── run_*.py
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

fstat.precompute_M(f0)

fstatz = fstat.compute_Fe(f0, skyloc)
```

The result `fstatz` contains the Fe-statistic evaluated at each sky location.

---

# Example Run Script

Files named

```
run_*.py
```

provide **example scripts showing how to run the Fe-statistic search**.

These scripts typically:

1. Load a set of pulsars
2. Inject a continuous GW signal
3. Compute the Fe-statistic
4. Produce a sky map of the statistic

They serve as **minimal working examples of the pipeline**.

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