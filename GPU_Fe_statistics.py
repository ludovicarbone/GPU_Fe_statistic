import os
import jax
import jax.numpy as jnp
import numpy as np


def fpc_fast(pos, gwtheta, gwphi):
    """
    Antenna pattern functions in jax.
    pos = x,y,z of where you want to compute it.
    """
    x, y, z = pos

    sin_phi   = jnp.sin(gwphi)
    cos_phi   = jnp.cos(gwphi)
    sin_theta = jnp.sin(gwtheta)
    cos_theta = jnp.cos(gwtheta)

    m_dot_pos     = sin_phi * x - cos_phi * y
    n_dot_pos     = -cos_theta * cos_phi * x - cos_theta * sin_phi * y + sin_theta * z
    omhat_dot_pos = -sin_theta * cos_phi * x - sin_theta * sin_phi * y - cos_theta * z

    denom  = 1.0 + omhat_dot_pos
    fplus  = 0.5 * (m_dot_pos**2 - n_dot_pos**2) / denom
    fcross = (m_dot_pos * n_dot_pos) / denom

    return fplus, fcross, -omhat_dot_pos


@jax.jit
def inner_product_jax(x, y, Ni_diag, T):
    """
    Replica of innerProduct_rr in JAX with marginalization over the timing model.

    Computes: x^T N^{-1} y - x^T N^{-1} T (T^T N^{-1} T)^{-1} T^T N^{-1} y

    :param x:       timeseries vector, shape (ntoa,)
    :param y:       timeseries vector, shape (ntoa,)
    :param Ni_diag: diagonal of N^{-1} = 1/toaerrs^2, shape (ntoa,)
    :param T:       timing model design matrix, shape (ntoa, npar)
    :return:        inner product (x|y)
    """
    xNy   = jnp.dot(x, Ni_diag * y)
    TNx   = jnp.dot(T.T, Ni_diag * x)
    TNy   = jnp.dot(T.T, Ni_diag * y)
    Sigma = jnp.dot(T.T * Ni_diag[None, :], T)
    return xNy - jnp.dot(TNx, jnp.linalg.solve(Sigma, TNy))


def get_template_inner_products(psrs, f0):
    """
    Computes the 3 inner products that depend ONLY on the sinusoidal templates
    and on the PTA noise (NOT on the residuals).

    These form the matrix M and can be reused for different injections
    at the same frequency f0.

    :param psrs: list of pulsars
    :param f0:   GW frequency
    :return:     array shape (n_psr, 3) with columns [sNs, cNc, sNc]
    """
    tref    = 53000 * 86400
    results = []

    for psr in psrs:
        toas    = jnp.array(psr.toas)
        cosine  = jnp.cos(2 * jnp.pi * f0 * (toas - tref))
        sine    = jnp.sin(2 * jnp.pi * f0 * (toas - tref))
        Ni_diag = 1.0 / jnp.array(psr.toaerrs)**2
        T       = jnp.array(psr.Mmat)

        sNs = inner_product_jax(sine,   sine,   Ni_diag, T)
        cNc = inner_product_jax(cosine, cosine, Ni_diag, T)
        sNc = inner_product_jax(sine,   cosine, Ni_diag, T)

        results.append(jnp.array([sNs, cNc, sNc]))

    return jnp.stack(results)   # (n_psr, 3)


def get_data_inner_products(psrs, f0):
    """
    Computes the 2 inner products that depend on the observed residuals.

    These form the vector N and must be recomputed every time
    the residuals change (e.g. different injection).

    :param psrs: list of pulsars
    :param f0:   GW frequency
    :return:     array shape (n_psr, 2) with columns [resNs, resNc]
    """
    tref    = 53000 * 86400
    results = []

    for psr in psrs:
        toas    = jnp.array(psr.toas)
        cosine  = jnp.cos(2 * jnp.pi * f0 * (toas - tref))
        sine    = jnp.sin(2 * jnp.pi * f0 * (toas - tref))
        res     = jnp.array(psr.residuals)
        Ni_diag = 1.0 / jnp.array(psr.toaerrs)**2
        T       = jnp.array(psr.Mmat)

        resNs = inner_product_jax(res, sine,   Ni_diag, T)
        resNc = inner_product_jax(res, cosine, Ni_diag, T)

        results.append(jnp.array([resNs, resNc]))

    return jnp.stack(results)   # (n_psr, 2)


def build_M_matrix(template_ips):
    """
    Builds the matrix M from the template inner products.

    :param template_ips: array (n_psr, 3) with columns [sNs, cNc, sNc]
    :return:             array (n_psr, 4, 4)
    """
    sNs = template_ips[:, 0]
    cNc = template_ips[:, 1]
    sNc = template_ips[:, 2]

    row1 = jnp.stack([sNs, sNc, sNs, sNc], axis=1)
    row2 = jnp.stack([sNc, cNc, sNc, cNc], axis=1)
    return jnp.stack([row1, row2, row1, row2], axis=2)   # (n_psr, 4, 4)


def build_N_vector(data_ips):
    """
    Builds the vector N from the data inner products.

    :param data_ips: array (n_psr, 2) with columns [resNs, resNc]
    :return:         array (n_psr, 4)
    """
    resNs = data_ips[:, 0]
    resNc = data_ips[:, 1]
    return jnp.stack([resNs, resNc, resNs, resNc], axis=1)   # (n_psr, 4)


class GPU_FeStat(object):

    def __init__(self, psrs, params=None, orf=None):
        print('Initializing the model...')
        self.psrs   = psrs
        self.params = params

        # Pulsar positions precomputed only once
        self.pos = jnp.array([psr.pos for psr in psrs])

        # fpc_vmap built only once
        self._fpc_vmap = jax.vmap(fpc_fast, in_axes=(0, None, None))

        # Cache for M: filled by precompute_M()
        self._M_cache = None
        self._M_f0    = None

    def precompute_M(self, f0):
        """
        Precomputes and stores the matrix M for a given frequency f0.

        Should be called when the PTA or f0 changes, but NOT when only
        the residuals change (different injection). The result is cached
        internally and automatically reused by compute_Fe.

        :param f0: GW frequency
        :return:   array (n_psr, 4, 4) — also returned for external use
        """
        print(f'Precomputing M matrix for f0={f0:.2e} Hz...')
        template_ips  = get_template_inner_products(self.psrs, f0)
        self._M_cache = build_M_matrix(template_ips)
        self._M_f0    = f0
        return self._M_cache

    def compute_Fe(self, f0, gw_skyloc, psr_theta_phi=None,
                   M_matrix=None, maximized_parameters=False):
        """
        Computes the Fe-statistic on a grid of sky positions.

        :param f0:                  GW frequency
        :param gw_skyloc:           array 2 x n_sky with [theta, phi] for each sky position
        :param psr_theta_phi:       custom pulsar positions (optional), shape (n_psr, 2)
        :param M_matrix:            precomputed M matrix (optional), shape (n_psr, 4, 4).
                                    If None, the internal cache is used (if available
                                    and computed at the same f0), otherwise recomputed.
        :param maximized_parameters: not implemented yet
        :return:                    array shape (n_sky,) with Fe-statistic values
        """
        # --- Resolve M ---
        if M_matrix is not None:
            # User passed M explicitly (e.g. computed externally)
            M = M_matrix
        elif self._M_cache is not None and self._M_f0 == f0:
            # Cache valid for this f0: reuse without recomputing
            M = self._M_cache
        else:
            # No cache available: compute on the fly and store
            if self._M_cache is not None:
                print(f'f0 changed ({self._M_f0:.2e} -> {f0:.2e} Hz): recomputing M...')
            M = self.precompute_M(f0)

        # --- Vector N from residuals (always recomputed) ---
        data_ips = get_data_inner_products(self.psrs, f0)
        N        = build_N_vector(data_ips)   # (n_psr, 4)

        # --- Pulsar positions ---
        if psr_theta_phi is not None:
            ptheta = psr_theta_phi[:, 0]
            pphi   = psr_theta_phi[:, 1]
            pos = jnp.stack([jnp.cos(pphi) * jnp.sin(ptheta),
                              jnp.sin(pphi) * jnp.sin(ptheta),
                              jnp.cos(ptheta)], axis=1)
        else:
            pos = self.pos

        fpc_psrs = self._fpc_vmap

        @jax.jit
        def fstat_sky_grid(gw_skyloc):
            def fstat_one_sky(gw_pos):
                F_p, F_c, _ = fpc_psrs(pos, gw_pos[0], gw_pos[1])

                F_stack = jnp.stack([F_p, F_p, F_c, F_c], axis=1)
                NN = N * F_stack

                Fp2  = F_p ** 2
                Fc2  = F_c ** 2
                FpFc = F_p * F_c
                rowA   = jnp.stack([Fp2,  Fp2,  FpFc, FpFc], axis=1)
                rowB   = jnp.stack([FpFc, FpFc, Fc2,  Fc2 ], axis=1)
                Mscale = jnp.stack([rowA, rowA, rowB, rowB], axis=2)
                MM = M * Mscale

                N_sum = jnp.sum(NN, axis=0)
                M_sum = jnp.sum(MM, axis=0)
                x = jnp.linalg.solve(M_sum, N_sum)
                return 0.5 * jnp.dot(N_sum, x)

            return jax.vmap(fstat_one_sky)(gw_skyloc.T)

        return fstat_sky_grid(gw_skyloc)