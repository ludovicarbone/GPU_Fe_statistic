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


# ── Inner product ─────────────────────────────────────────────────────────────

@jax.jit
def inner_product_jax(x, y, Ni_diag, T, Sigma):
    """
    Computes the inner product with marginalisation over timing model + GP background.

    (x|y) = x^T N^{-1} y  -  (T^T N^{-1} x)^T  Sigma^{-1}  (T^T N^{-1} y)

    where Sigma = T^T N^{-1} T + Phi^{-1} includes both timing model and
    (optionally) CRN.

    When background=False the caller simply passes T = Mmat and
    Sigma = Mmat^T N^{-1} Mmat, reproducing the original white-noise behaviour.

    :param x:       timeseries vector, shape (ntoa,)
    :param y:       timeseries vector, shape (ntoa,)
    :param Ni_diag: diagonal of N^{-1} = 1/toaerrs^2, shape (ntoa,)
    :param T:       basis matrix [T_tm | F_crn] or just Mmat, shape (ntoa, n_basis)
    :param Sigma:   T^T N^{-1} T + Phi^{-1}, shape (n_basis, n_basis)
    :return:        scalar inner product (x|y)
    """
    xNy = jnp.dot(x, Ni_diag * y)
    TNx = jnp.dot(T.T, Ni_diag * x)
    TNy = jnp.dot(T.T, Ni_diag * y)
    solution, _, _, _ = jnp.linalg.lstsq(Sigma, TNy, rcond=1e-10)
    return xNy - jnp.dot(TNx, solution)


# ── Per-pulsar Sigma construction (background ON) ────────────────────────────

def build_per_pulsar_sigma(lik, params, n_crn=60):
    """
    Builds per-pulsar (T_tot, Sigma, Ni_diag) including timing model + CRN
    (block-diagonal approximation of the full cross-correlated Phi_inv).

    :param lik:    GlobalLikelihood with background (discovery)
    :param params: dict with background parameters
    :param n_crn:  number of CRN basis functions per pulsar (default 60)
    :return:       list of (T_tot, Sigma, Ni_diag) tuples, one per pulsar
    """
    n_psr = len(lik.psls)

    P_var_inv    = lik.globalgp.Phi_inv or lik.globalgp.Phi.make_inv()
    Pinv_crn, _  = P_var_inv(params)

    results = []
    for i, psl in enumerate(lik.psls):
        Ni_diag = 1.0 / jnp.array(psl.N.N.N)

        F_tm  = jnp.array(psl.N.F)
        F_crn = jnp.array(lik.globalgp.Fs[i])

        T_tot = jnp.concatenate([F_tm, F_crn], axis=1)

        FtNmF = jnp.dot(T_tot.T * Ni_diag[None, :], T_tot)

        Phi_inv_tm = jnp.diag(jnp.array(psl.N.P.N))

        Phi_inv_crn_ii = jnp.array(
            Pinv_crn[i * n_crn:(i + 1) * n_crn,
                     i * n_crn:(i + 1) * n_crn]
        )

        n_tm    = F_tm.shape[1]
        Phi_inv = jnp.zeros((n_tm + n_crn, n_tm + n_crn))
        Phi_inv = Phi_inv.at[:n_tm, :n_tm].set(Phi_inv_tm)
        Phi_inv = Phi_inv.at[n_tm:, n_tm:].set(Phi_inv_crn_ii)

        Sigma = FtNmF + Phi_inv

        results.append((T_tot, Sigma, Ni_diag))

    return results


# ── Per-pulsar noise for white-noise-only mode ───────────────────────────────

def build_per_pulsar_white_noise(psrs):
    """
    Builds per-pulsar (T, Sigma, Ni_diag) using only the timing design matrix
    and white noise — no background GP.

    :param psrs: list of pulsars
    :return:     list of (T, Sigma, Ni_diag) tuples
    """
    results = []
    for psr in psrs:
        T       = jnp.array(psr.Mmat)
        Ni_diag = 1.0 / jnp.array(psr.toaerrs) ** 2
        Sigma   = jnp.dot(T.T * Ni_diag[None, :], T)
        results.append((T, Sigma, Ni_diag))
    return results


# ── Template and data inner products ──────────────────────────────────────────

def get_template_inner_products(psrs, f0, per_psr_noise):
    """
    Computes the 3 inner products that depend ONLY on the sinusoidal templates.

    :param psrs:          list of pulsars
    :param f0:            GW frequency
    :param per_psr_noise: list of (T, Sigma, Ni_diag)
    :return:              array shape (n_psr, 3) with columns [sNs, cNc, sNc]
    """
    tref    = 53000 * 86400
    results = []

    for psr, (T_tot, Sigma, Ni_diag) in zip(psrs, per_psr_noise):
        toas   = jnp.array(psr.toas)
        cosine = jnp.cos(2 * jnp.pi * f0 * (toas - tref))
        sine   = jnp.sin(2 * jnp.pi * f0 * (toas - tref))

        sNs = inner_product_jax(sine,   sine,   Ni_diag, T_tot, Sigma)
        cNc = inner_product_jax(cosine, cosine, Ni_diag, T_tot, Sigma)
        sNc = inner_product_jax(sine,   cosine, Ni_diag, T_tot, Sigma)

        results.append(jnp.array([sNs, cNc, sNc]))

    return jnp.stack(results)


def get_data_inner_products(psrs, f0, per_psr_noise):
    """
    Computes the 2 inner products that depend on the observed residuals.

    :param psrs:          list of pulsars
    :param f0:            GW frequency
    :param per_psr_noise: list of (T, Sigma, Ni_diag)
    :return:              array shape (n_psr, 2) with columns [resNs, resNc]
    """
    tref    = 53000 * 86400
    results = []

    for psr, (T_tot, Sigma, Ni_diag) in zip(psrs, per_psr_noise):
        toas   = jnp.array(psr.toas)
        cosine = jnp.cos(2 * jnp.pi * f0 * (toas - tref))
        sine   = jnp.sin(2 * jnp.pi * f0 * (toas - tref))
        res    = jnp.array(psr.residuals)

        resNs = inner_product_jax(res, sine,   Ni_diag, T_tot, Sigma)
        resNc = inner_product_jax(res, cosine, Ni_diag, T_tot, Sigma)

        results.append(jnp.array([resNs, resNc]))

    return jnp.stack(results)


def get_data_inner_products_from_residuals(residuals, psrs, f0, per_psr_noise):
    """
    Like get_data_inner_products but takes explicit residual arrays
    instead of reading psr.residuals.

    :param residuals:     list of arrays (n_psr,), each shape (n_toa_i,)
    :param psrs:          list of pulsars
    :param f0:            GW frequency
    :param per_psr_noise: list of (T, Sigma, Ni_diag)
    :return:              array shape (n_psr, 2) with columns [resNs, resNc]
    """
    tref    = 53000 * 86400
    results = []

    for i, (psr, (T_tot, Sigma, Ni_diag)) in enumerate(zip(psrs, per_psr_noise)):
        toas   = jnp.array(psr.toas)
        res    = jnp.array(residuals[i])
        cosine = jnp.cos(2 * jnp.pi * f0 * (toas - tref))
        sine   = jnp.sin(2 * jnp.pi * f0 * (toas - tref))

        resNs = inner_product_jax(res, sine,   Ni_diag, T_tot, Sigma)
        resNc = inner_product_jax(res, cosine, Ni_diag, T_tot, Sigma)

        results.append(jnp.array([resNs, resNc]))

    return jnp.stack(results)


# ── M matrix and N vector ─────────────────────────────────────────────────────

def build_M_matrix(template_ips):
    sNs = template_ips[:, 0]
    cNc = template_ips[:, 1]
    sNc = template_ips[:, 2]
    row1 = jnp.stack([sNs, sNc, sNs, sNc], axis=1)
    row2 = jnp.stack([sNc, cNc, sNc, cNc], axis=1)
    return jnp.stack([row1, row2, row1, row2], axis=2)


def build_N_vector(data_ips):
    resNs = data_ips[:, 0]
    resNc = data_ips[:, 1]
    return jnp.stack([resNs, resNc, resNs, resNc], axis=1)


# ── Main class ────────────────────────────────────────────────────────────────

class GPU_FeStat(object):

    def __init__(self, psrs, lik=None, params=None, n_crn=60):
        """
        :param psrs:   list of pulsars
        :param lik:    GlobalLikelihood from discovery (with background).
                       If None or if lik.globalgp is None, falls back to
                       white noise + timing model only.
        :param params: dict of background parameters,
                       e.g. {'crn_log10_A': -15, 'crn_gamma': 4.33}
        :param n_crn:  CRN basis functions per pulsar (default 60)
        """
        print('Initializing the model...')
        self.psrs   = psrs
        self.lik    = lik
        self.params = params
        self.n_crn  = n_crn

        self.pos       = jnp.array([psr.pos for psr in psrs])
        self._fpc_vmap = jax.vmap(fpc_fast, in_axes=(0, None, None))
        self._M_cache  = None
        self._M_f0     = None

        # ── CHANGED: guard also on lik.globalgp being not None ────────────
        has_background = (lik is not None
                          and params is not None
                          and getattr(lik, 'globalgp', None) is not None)

        if has_background:
            print('Building per-pulsar Sigma matrices (timing + CRN)...')
            self._per_psr_noise = build_per_pulsar_sigma(lik, params, n_crn=n_crn)
        else:
            print('No background: using white noise + timing model only.')
            self._per_psr_noise = build_per_pulsar_white_noise(psrs)

    def update_params(self, params):
        """
        Recompute Sigma matrices when background parameters change.
        Also invalidates the M cache.
        """
        self.params = params
        if self.lik is not None and getattr(self.lik, 'globalgp', None) is not None:
            print('Updating per-pulsar Sigma matrices...')
            self._per_psr_noise = build_per_pulsar_sigma(self.lik, params, n_crn=self.n_crn)
            self._M_cache = None
            self._M_f0    = None

    def precompute_M(self, f0):
        print(f'Precomputing M matrix for f0={f0:.2e} Hz...')
        template_ips  = get_template_inner_products(self.psrs, f0, self._per_psr_noise)
        self._M_cache = build_M_matrix(template_ips)
        self._M_f0    = f0
        return self._M_cache

    def compute_Fe(self, f0, gw_skyloc, psr_theta_phi=None, M_matrix=None):
        """
        Computes the Fe-statistic on a grid of sky positions.

        :param f0:            GW frequency
        :param gw_skyloc:     array (2, n_sky) with [theta, phi]
        :param psr_theta_phi: custom pulsar positions (optional), shape (n_psr, 2)
        :param M_matrix:      precomputed M matrix (optional)
        :return:              array shape (n_sky,)
        """
        if M_matrix is not None:
            M = M_matrix
        elif self._M_cache is not None and self._M_f0 == f0:
            M = self._M_cache
        else:
            if self._M_cache is not None:
                print(f'f0 changed ({self._M_f0:.2e} -> {f0:.2e} Hz): recomputing M...')
            M = self.precompute_M(f0)

        data_ips = get_data_inner_products(self.psrs, f0, self._per_psr_noise)
        N        = build_N_vector(data_ips)

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

        Parameters
        ----------
        f0              : float — GW frequency
        gw_skyloc       : array shape (2, n_sky)
        residuals_batch : list of arrays, one per pulsar, each shape (N, n_toa_i)

        Returns
        -------
        fe_maps : jax array shape (N, n_sky)
        """
        return _compute_Fe_batch_impl(
            residuals_batch = residuals_batch,
            psrs            = self.psrs,
            pos             = self.pos,
            fpc_psrs        = self._fpc_vmap,
            per_psr_noise   = self._per_psr_noise,
            f0              = f0,
            gw_skyloc       = gw_skyloc,
        )


# ── Batch implementation ─────────────────────────────────────────────────────

def _compute_Fe_batch_impl(residuals_batch, psrs, pos, fpc_psrs,
                           per_psr_noise, f0, gw_skyloc):
    """
    Core vmapped implementation.  Uses per_psr_noise (which already encodes
    whether the background is on or off) so the logic is identical for both
    modes.
    """

    # ── M matrix: computed once ───────────────────────────────────────────────
    template_ips = get_template_inner_products(psrs, f0, per_psr_noise)
    M = build_M_matrix(template_ips)

    # ── Pre-extract per-pulsar arrays for the vmap closure ────────────────────
    tref = 53000 * 86400
    sines, cosines = [], []
    T_list, Sigma_list, Ni_list = [], [], []

    for psr, (T_tot, Sigma, Ni_diag) in zip(psrs, per_psr_noise):
        toas = jnp.array(psr.toas)
        sines.append(jnp.sin(2 * jnp.pi * f0 * (toas - tref)))
        cosines.append(jnp.cos(2 * jnp.pi * f0 * (toas - tref)))
        T_list.append(T_tot)
        Sigma_list.append(Sigma)
        Ni_list.append(Ni_diag)

    n_psr = len(psrs)

    # ── Step 1: vmap data inner products over N injections ────────────────────
    def get_N_vec_one(res_list):
        ips = []
        for i in range(n_psr):
            resNs = inner_product_jax(res_list[i], sines[i],   Ni_list[i], T_list[i], Sigma_list[i])
            resNc = inner_product_jax(res_list[i], cosines[i], Ni_list[i], T_list[i], Sigma_list[i])
            ips.append(jnp.array([resNs, resNc]))
        return build_N_vector(jnp.stack(ips))

    N_vecs = jax.vmap(get_N_vec_one)(residuals_batch)

    # ── Step 2: vmap Fe map computation over N injections ─────────────────────
    @jax.jit
    def all_fe_maps(N_vecs):
        def fe_map_one_injection(N_vec):
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

            return jax.vmap(fstat_one_sky)(gw_skyloc.T)

        return jax.vmap(fe_map_one_injection)(N_vecs)

    return all_fe_maps(N_vecs)


def compute_Fe_batch(residuals_batch, psrs, f0, gw_skyloc,
                     lik=None, params=None, n_crn=60):
    """
    Standalone convenience wrapper — no GPU_FeStat instance needed.

    When lik and params are given the CRN background is marginalised;
    otherwise only white noise + timing model are used.
    """
    pos      = jnp.array([psr.pos for psr in psrs])
    fpc_psrs = jax.vmap(fpc_fast, in_axes=(0, None, None))

    has_background = (lik is not None
                      and params is not None
                      and getattr(lik, 'globalgp', None) is not None)

    if has_background:
        per_psr_noise = build_per_pulsar_sigma(lik, params, n_crn=n_crn)
    else:
        per_psr_noise = build_per_pulsar_white_noise(psrs)

    return _compute_Fe_batch_impl(residuals_batch, psrs, pos, fpc_psrs,
                                  per_psr_noise, f0, gw_skyloc)


def compute_Fe_from_residuals(residuals, psrs, f0, gw_skyloc,
                              lik=None, params=None, n_crn=60):
    """
    One-shot Fe-statistic from explicit residual arrays (no GPU_FeStat needed).
    """
    pos      = jnp.array([psr.pos for psr in psrs])
    fpc_psrs = jax.vmap(fpc_fast, in_axes=(0, None, None))

    has_background = (lik is not None
                      and params is not None
                      and getattr(lik, 'globalgp', None) is not None)

    if has_background:
        per_psr_noise = build_per_pulsar_sigma(lik, params, n_crn=n_crn)
    else:
        per_psr_noise = build_per_pulsar_white_noise(psrs)

    template_ips = get_template_inner_products(psrs, f0, per_psr_noise)
    M = build_M_matrix(template_ips)

    data_ips = get_data_inner_products_from_residuals(residuals, psrs, f0, per_psr_noise)
    N = build_N_vector(data_ips)

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