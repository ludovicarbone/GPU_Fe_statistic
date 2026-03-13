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
    # Rimuovi colonne zero da T
    col_norms = jnp.sqrt(jnp.sum(T**2, axis=0))
    # Sostituisci colonne zero con colonne identità per stabilità
    T = jnp.where(col_norms[None, :] < 1e-10, 0.0, T)

    xNy   = jnp.dot(x, Ni_diag * y)
    TNx   = jnp.dot(T.T, Ni_diag * x)
    TNy   = jnp.dot(T.T, Ni_diag * y)
    Sigma = jnp.dot(T.T * Ni_diag[None, :], T)

    # Usa lstsq invece di solve — robusto a matrici singolari
    solution, _, _, _ = jnp.linalg.lstsq(Sigma, TNy, rcond=1e-10)
    return xNy - jnp.dot(TNx, solution)


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

        self.pos = jnp.array([psr.pos for psr in psrs])
        self._fpc_vmap = jax.vmap(fpc_fast, in_axes=(0, None, None))

    def compute_Fe(self, f0, gw_skyloc, psr_theta_phi=None,
                   maximized_parameters=False):
        """
        Computes the Fe-statistic on a grid of sky positions.

        :param f0:                   GW frequency
        :param gw_skyloc:            array 2 x n_sky with [theta, phi]
        :param psr_theta_phi:        custom pulsar positions (optional), shape (n_psr, 2)
        :param maximized_parameters: not implemented yet
        :return:                     array shape (n_sky,) with Fe-statistic values
        """
        template_ips = get_template_inner_products(self.psrs, f0)
        M            = build_M_matrix(template_ips)   # (n_psr, 4, 4)

        data_ips = get_data_inner_products(self.psrs, f0)
        N        = build_N_vector(data_ips)            # (n_psr, 4)

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

    def compute_Fe_batch(self, f0, gw_skyloc, residuals_batch):
        """
        Computes Fe-statistic maps for N injections in a single vmapped pass.

        The M matrix (template inner products) is shared across all injections
        and computed only once. Only the N vector (data inner products) is
        recomputed per injection, via vmap.

        Parameters
        ----------
        f0              : float
            GW frequency.
        gw_skyloc       : array, shape (2, n_sky)
            Sky grid [theta, phi] in healpy convention.
        residuals_batch : list of arrays, one per pulsar, each shape (N, n_toa_i)
            Output of generate_residual_samples_vmap, i.e. residuals_chunk from
            the sampler vmap. residuals_batch[i][j] = residuals of pulsar i,
            injection j.

        Returns
        -------
        fe_maps : jax array, shape (N, n_sky)
            Fe-statistic map for each of the N injections.

        Example
        -------
        # residuals is the output of generate_residual_samples_vmap
        # for a single chunk; residuals[0] has shape (N, n_toa_i) per pulsar
        fe_maps = fstat.compute_Fe_batch(f0, skyloc, residuals[0])
        """
        return _compute_Fe_batch_impl(
            residuals_batch = residuals_batch,
            psrs            = self.psrs,
            pos             = self.pos,
            fpc_psrs        = self._fpc_vmap,
            f0              = f0,
            gw_skyloc       = gw_skyloc,
        )


# ── batch implementation (standalone, usable without GPU_FeStat) ──────────────

def _compute_Fe_batch_impl(residuals_batch, psrs, pos, fpc_psrs, f0, gw_skyloc):
    """
    Core vmapped implementation shared by GPU_FeStat.compute_Fe_batch
    and the standalone compute_Fe_batch function.

    Separation between:
      • M matrix  — template inner products, computed ONCE, independent of residuals
      • N vectors — data inner products, vmapped over N injections
      • Fe maps   — sky-grid computation, vmapped over N injections
    """

    # ── M matrix: computed once ────────────────────────────────────────────────
    template_ips = get_template_inner_products(psrs, f0)
    M = build_M_matrix(template_ips)   # (n_psr, 4, 4)

    # ── Per-pulsar noise + templates: precomputed, captured in closure ─────────
    tref = 53000 * 86400
    sines, cosines, Ni_diags, Tmats = [], [], [], []
    for psr in psrs:
        toas    = jnp.array(psr.toas)
        Ni_diag = 1.0 / jnp.array(psr.toaerrs) ** 2
        T       = jnp.array(psr.Mmat)
        sines.append(  jnp.sin(2 * jnp.pi * f0 * (toas - tref)))
        cosines.append(jnp.cos(2 * jnp.pi * f0 * (toas - tref)))
        Ni_diags.append(Ni_diag)
        Tmats.append(T)

    n_psr = len(psrs)

    # ── Step 1: vmap data inner products over N injections ────────────────────
    # residuals_batch is a list/tuple of (N, n_toa_i) arrays, one per pulsar.
    # jax.vmap broadcasts axis-0 of every leaf in the pytree, so each call to
    # get_N_vec_one sees a list of (n_toa_i,) arrays — one injection at a time.

    def get_N_vec_one(res_list):
        """
        res_list : list of (n_toa_i,) arrays  — residuals for ONE injection.
        returns  : (n_psr, 4) N vector.
        """
        ips = []
        for i in range(n_psr):
            resNs = inner_product_jax(res_list[i], sines[i],   Ni_diags[i], Tmats[i])
            resNc = inner_product_jax(res_list[i], cosines[i], Ni_diags[i], Tmats[i])
            ips.append(jnp.array([resNs, resNc]))
        return build_N_vector(jnp.stack(ips))   # (n_psr, 4)

    # vmap over the N (batch) dimension — axis 0 of every array in residuals_batch
    N_vecs = jax.vmap(get_N_vec_one)(residuals_batch)   # (N, n_psr, 4)

    # ── Step 2: vmap Fe map computation over N injections ────────────────────
    # Inner vmap: over sky positions.  Outer vmap: over injections.

    @jax.jit
    def all_fe_maps(N_vecs):

        def fe_map_one_injection(N_vec):
            """
            N_vec   : (n_psr, 4)  — N vector for ONE injection.
            returns : (n_sky,)    — Fe map.
            """
            def fstat_one_sky(gw_pos):
                F_p, F_c, _ = fpc_psrs(pos, gw_pos[0], gw_pos[1])

                F_stack = jnp.stack([F_p, F_p, F_c, F_c], axis=1)
                NN = N_vec * F_stack

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

            # inner vmap: sky positions
            return jax.vmap(fstat_one_sky)(gw_skyloc.T)   # (n_sky,)

        # outer vmap: injections
        return jax.vmap(fe_map_one_injection)(N_vecs)      # (N, n_sky)

    return all_fe_maps(N_vecs)


def compute_Fe_batch(residuals_batch, psrs, f0, gw_skyloc):
    """
    Standalone convenience wrapper — no GPU_FeStat instance needed.

    Parameters
    ----------
    residuals_batch : list of arrays, one per pulsar, each shape (N, n_toa_i)
    psrs            : list of pulsars
    f0              : float, GW frequency
    gw_skyloc       : array shape (2, n_sky)

    Returns
    -------
    fe_maps : jax array shape (N, n_sky)
    """
    pos      = jnp.array([psr.pos for psr in psrs])
    fpc_psrs = jax.vmap(fpc_fast, in_axes=(0, None, None))
    return _compute_Fe_batch_impl(residuals_batch, psrs, pos, fpc_psrs, f0, gw_skyloc)


# ── existing helpers (unchanged) ──────────────────────────────────────────────

def get_data_inner_products_from_residuals(residuals, psrs, f0):
    """
    Compute the data inner products for a given set of residuals.

    residuals: list of arrays (n_psr,), each shape (n_toa_i,)
    """
    tref = 53000 * 86400
    results = []

    for i, psr in enumerate(psrs):
        toas    = jnp.array(psr.toas)
        res     = jnp.array(residuals[i])
        Ni_diag = 1.0 / jnp.array(psr.toaerrs)**2
        T       = jnp.array(psr.Mmat)

        cosine = jnp.cos(2 * jnp.pi * f0 * (toas - tref))
        sine   = jnp.sin(2 * jnp.pi * f0 * (toas - tref))

        resNs = inner_product_jax(res, sine,   Ni_diag, T)
        resNc = inner_product_jax(res, cosine, Ni_diag, T)

        results.append(jnp.array([resNs, resNc]))

    return jnp.stack(results)   # (n_psr, 2)


def compute_Fe_from_residuals(residuals, psrs, f0, gw_skyloc):

    pos = jnp.array([psr.pos for psr in psrs])

    template_ips = get_template_inner_products(psrs, f0)
    M = build_M_matrix(template_ips)

    data_ips = get_data_inner_products_from_residuals(residuals, psrs, f0)
    N = build_N_vector(data_ips)

    fpc_psrs = jax.vmap(fpc_fast, in_axes=(0, None, None))

    def fstat_one_sky(gw_pos):

        F_p, F_c, _ = fpc_psrs(pos, gw_pos[0], gw_pos[1])

        F_stack = jnp.stack([F_p, F_p, F_c, F_c], axis=1)
        NN = N * F_stack

        Fp2  = F_p**2
        Fc2  = F_c**2
        FpFc = F_p*F_c

        rowA = jnp.stack([Fp2,  Fp2,  FpFc, FpFc], axis=1)
        rowB = jnp.stack([FpFc, FpFc, Fc2,  Fc2 ], axis=1)

        Mscale = jnp.stack([rowA, rowA, rowB, rowB], axis=2)
        MM = M * Mscale

        N_sum = jnp.sum(NN, axis=0)
        M_sum = jnp.sum(MM, axis=0)

        x = jnp.linalg.solve(M_sum, N_sum)

        return 0.5 * jnp.dot(N_sum, x)

    return jax.vmap(fstat_one_sky)(gw_skyloc.T)