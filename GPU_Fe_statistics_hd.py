import os
import jax
import jax.numpy as jnp
import numpy as np
from per_pulsar_noise import build_per_pulsar_noise
import jax.scipy.linalg as jsl
from per_pulsar_noise import unpack_pulsar_kernel, crn_block_slices


def compute_tref(psrs):
    """
    Data-centered reference epoch (seconds): mean over all TOAs of all pulsars.
    MUST match the tref used by the injection (injection_discovery.compute_tref).
    """
    return float(np.mean(np.concatenate([np.asarray(p.toas) for p in psrs])))


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
    Computes the inner product with marginalisation over timing model + GP noise.

        (x|y) = x^T N^{-1} y  -  (T^T N^{-1} x)^T  Sigma^{-1}  (T^T N^{-1} y)

    where Sigma = T^T N^{-1} T + Phi^{-1} includes the timing model and,
    optionally, per-pulsar GPs (red noise, DM) and the CRN background. All the
    block structure is encoded upstream in (T, Sigma) by build_per_pulsar_noise;
    this routine is agnostic to which components are present.

    NOTE: the linear system is solved with jnp.linalg.solve, NOT lstsq. With
    lstsq(rcond) the singular values of an ill-conditioned Sigma (e.g. the raw
    Mmat basis, cond ~ 1e42) get truncated, leaving timing-model power in the
    statistic and inflating the Fe at low frequency bins. build_per_pulsar_noise
    uses the orthonormal SVD timing basis (cond ~ 1), so solve is both correct
    and well-posed here.

    :param x:       timeseries vector, shape (ntoa,)
    :param y:       timeseries vector, shape (ntoa,)
    :param Ni_diag: diagonal of N^{-1} = 1/toaerrs^2, shape (ntoa,)
    :param T:       basis matrix [T_tm | F_gp | F_crn], shape (ntoa, n_basis)
    :param Sigma:   T^T N^{-1} T + Phi^{-1}, shape (n_basis, n_basis)
    :return:        scalar inner product (x|y)
    """
    xNy = jnp.dot(x, Ni_diag * y)
    TNx = jnp.dot(T.T, Ni_diag * x)
    TNy = jnp.dot(T.T, Ni_diag * y)
    solution = jnp.linalg.solve(Sigma, TNy)
    return xNy - jnp.dot(TNx, solution)


# ── Template and data inner products ──────────────────────────────────────────

def get_template_inner_products(psrs, f0, per_psr_noise, tref=None):
    """
    Computes the 3 inner products that depend ONLY on the sinusoidal templates.

    :param psrs:          list of pulsars
    :param f0:            GW frequency
    :param per_psr_noise: list of (T, Sigma, Ni_diag)
    :return:              array shape (n_psr, 3) with columns [sNs, cNc, sNc]
    """
    if tref is None:
        tref = compute_tref(psrs)
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


def get_data_inner_products(psrs, f0, per_psr_noise, tref=None):
    """
    Computes the 2 inner products that depend on the observed residuals.

    :param psrs:          list of pulsars
    :param f0:            GW frequency
    :param per_psr_noise: list of (T, Sigma, Ni_diag)
    :return:              array shape (n_psr, 2) with columns [resNs, resNc]
    """
    if tref is None:
        tref = compute_tref(psrs)
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


def get_data_inner_products_from_residuals(residuals, psrs, f0, per_psr_noise, tref=None):
    """
    Like get_data_inner_products but takes explicit residual arrays
    instead of reading psr.residuals.

    :param residuals:     list of arrays (n_psr,), each shape (n_toa_i,)
    :param psrs:          list of pulsars
    :param f0:            GW frequency
    :param per_psr_noise: list of (T, Sigma, Ni_diag)
    :return:              array shape (n_psr, 2) with columns [resNs, resNc]
    """
    if tref is None:
        tref = compute_tref(psrs)
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

class GPU_FeStat_BlockDiag(object):

    def __init__(self, psrs, lik=None, params=None, n_crn=60, tref=None):
        """
        :param psrs:   list of pulsars
        :param lik:    GlobalLikelihood from discovery. The Fe noise model MUST
                       match the one used for the injection. If it carries a
                       globalgp and params are given, the CRN is marginalised;
                       if it carries per-pulsar GPs (red noise / DM), those are
                       marginalised too. Otherwise only white noise + timing.
        :param params: dict of noise parameters. Required whenever lik contains
                       a CRN background or per-pulsar GPs (build_per_pulsar_noise
                       raises if it is missing, instead of silently dropping the
                       block).
        :param n_crn:  kept for backward compatibility; the CRN block offsets are
                       now inferred from globalgp.Fs shapes and n_crn is unused.
        """
        print('Initializing the model...')
        self.psrs   = psrs
        self.tref   = compute_tref(psrs) if tref is None else tref
        print(f'Using tref = {self.tref:.6e} s')
        self.lik    = lik
        self.params = params
        self.n_crn  = n_crn

        self.pos       = jnp.array([psr.pos for psr in psrs])
        self._fpc_vmap = jax.vmap(fpc_fast, in_axes=(0, None, None))
        self._M_cache  = None
        self._M_f0     = None

        if lik is None:
            raise ValueError('GPU_FeStat richiede una likelihood: il modello '
                             'di rumore della Fe deve coincidere con quello '
                             'usato per l iniezione.')

        print('Building per-pulsar Sigma matrices...')
        self._per_psr_noise = build_per_pulsar_noise(lik, params)

    def update_params(self, params):
        """
        Recompute Sigma matrices when noise parameters change (CRN and/or
        per-pulsar red noise / DM). Also invalidates the M cache.
        """
        self.params = params
        self._per_psr_noise = build_per_pulsar_noise(self.lik, params)
        self._M_cache = None
        self._M_f0    = None

    def precompute_M(self, f0):
        print(f'Precomputing M matrix for f0={f0:.2e} Hz...')
        template_ips  = get_template_inner_products(self.psrs, f0, self._per_psr_noise, self.tref)
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

        data_ips = get_data_inner_products(self.psrs, f0, self._per_psr_noise, self.tref)
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
            tref            = self.tref,
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
                           per_psr_noise, f0, gw_skyloc, tref=None):
    """
    Core vmapped implementation.  Uses per_psr_noise (which already encodes
    which noise components are present) so the logic is identical for every
    mode: white-only, CRN background, per-pulsar red noise / DM, or any mix.
    """

    # ── M matrix: computed once ───────────────────────────────────────────────
    if tref is None:
        tref = compute_tref(psrs)

    template_ips = get_template_inner_products(psrs, f0, per_psr_noise, tref)
    M = build_M_matrix(template_ips)

    # ── Pre-extract per-pulsar arrays for the vmap closure ────────────────────
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


# ── Standalone convenience wrappers (no GPU_FeStat instance needed) ───────────

def compute_Fe_batch(residuals_batch, psrs, f0, gw_skyloc,
                     lik=None, params=None, n_crn=60, tref=None):
    """
    Standalone wrapper. The noise model is built by build_per_pulsar_noise from
    lik, so it handles white-only, CRN, and per-pulsar GP modes identically to
    GPU_FeStat. lik is required (the Fe noise model must match the injection).
    """
    if lik is None:
        raise ValueError('compute_Fe_batch richiede una likelihood: il modello '
                         'di rumore della Fe deve coincidere con quello usato '
                         'per l iniezione.')

    pos      = jnp.array([psr.pos for psr in psrs])
    fpc_psrs = jax.vmap(fpc_fast, in_axes=(0, None, None))

    per_psr_noise = build_per_pulsar_noise(lik, params)

    return _compute_Fe_batch_impl(residuals_batch, psrs, pos, fpc_psrs,
                                  per_psr_noise, f0, gw_skyloc, tref)


def compute_Fe_from_residuals(residuals, psrs, f0, gw_skyloc,
                              lik=None, params=None, n_crn=60, tref=None):
    """
    One-shot Fe-statistic from explicit residual arrays (no GPU_FeStat needed).
    Same noise handling as compute_Fe_batch.
    """
    if lik is None:
        raise ValueError('compute_Fe_from_residuals richiede una likelihood: il '
                         'modello di rumore della Fe deve coincidere con quello '
                         'usato per l iniezione.')

    pos      = jnp.array([psr.pos for psr in psrs])
    fpc_psrs = jax.vmap(fpc_fast, in_axes=(0, None, None))

    per_psr_noise = build_per_pulsar_noise(lik, params)

    if tref is None:
        tref = compute_tref(psrs)

    template_ips = get_template_inner_products(psrs, f0, per_psr_noise, tref)
    M = build_M_matrix(template_ips)

    data_ips = get_data_inner_products_from_residuals(residuals, psrs, f0, per_psr_noise, tref)
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


# ═══ Fe con ORF completa (HD) nel rumore ═══

# ── Rumore globale ───────────────────────────────────────────────────────────

def build_global_noise(lik, params, marginalize_timing=True, verbose=True):
    """
    Costruisce Sigma globale (con ORF completa) e le info per pulsar.

    :return: dict con
        'Sigma_lu' : fattorizzazione LU di Sigma (n_glob x n_glob)
        'n_glob'   : dimensione globale
        'psr'      : lista di dict per pulsar con Ni, T (ntoa, n_i), idx (n_i,)
                     dove idx sono gli indici globali delle colonne di T
    """
    globalgp = getattr(lik, 'globalgp', None)
    if globalgp is None:
        raise ValueError('La likelihood non ha un globalgp: usa GPU_FeStat '
                         'standard (senza CRN non ci sono correlazioni).')
    if params is None:
        raise ValueError('Servono i params del CRN (e di eventuali GP).')

    P_glob_inv = globalgp.Phi_inv or globalgp.Phi.make_inv()
    Pinv_crn, _ = P_glob_inv(params)
    Pinv_crn = jnp.asarray(Pinv_crn)
    crn_sl = crn_block_slices(globalgp)
    n_crn_tot = crn_sl[-1].stop

    # 1) blocchi locali (timing + GP per pulsar)
    locs, off = [], 0
    for psl in lik.psls:
        parts = unpack_pulsar_kernel(psl)
        n_tm = parts['F_tm'].shape[1]
        F = [jnp.asarray(parts['F_tm'])]
        if marginalize_timing:
            P = [jnp.zeros((n_tm, n_tm))]
        else:
            P = [jnp.diag(1.0 / jnp.asarray(psl.N.P.N))]
        if parts['P_var'] is not None:
            missing = [p for p in parts['P_var'].params if p not in params]
            if missing:
                raise KeyError(f'Parametri mancanti in params: {missing}')
            Pinv_gp, _ = parts['P_var'].make_inv()(params)
            F.append(jnp.asarray(parts['F_gp']))
            P.append(jnp.asarray(Pinv_gp))
        T_loc = jnp.concatenate(F, axis=1)
        Pinv_loc = jsl.block_diag(*P)
        n = T_loc.shape[1]
        locs.append((parts['Ni_diag'], T_loc, Pinv_loc, np.arange(off, off + n)))
        off += n
    n_loc_tot = off
    n_glob = n_loc_tot + n_crn_tot

    # 2) Sigma globale
    Sigma = jnp.zeros((n_glob, n_glob))
    psr_info = []
    for i, (Ni, T_loc, Pinv_loc, idx_loc) in enumerate(locs):
        F_crn = jnp.asarray(globalgp.Fs[i])
        idx_crn = n_loc_tot + np.arange(crn_sl[i].start, crn_sl[i].stop)
        T_i = jnp.concatenate([T_loc, F_crn], axis=1)
        idx_i = np.concatenate([idx_loc, idx_crn])

        TNT = jnp.dot(T_i.T * Ni[None, :], T_i)
        Sigma = Sigma.at[np.ix_(idx_i, idx_i)].add(TNT)
        Sigma = Sigma.at[np.ix_(idx_loc, idx_loc)].add(Pinv_loc)
        psr_info.append({'Ni': Ni, 'T': T_i, 'idx': idx_i})

    crn_all = np.arange(n_loc_tot, n_glob)
    Sigma = Sigma.at[np.ix_(crn_all, crn_all)].add(Pinv_crn)   # ORF piena qui

    if verbose:
        print(f'  Sigma globale: {n_glob} x {n_glob} '
              f'({n_loc_tot} locali + {n_crn_tot} CRN correlati)')

    return {'Sigma_lu': jsl.lu_factor(Sigma), 'n_glob': n_glob, 'psr': psr_info}


# ── Precalcolo dipendente da f0 ──────────────────────────────────────────────

def _templates(psrs, f0, tref):
    s, c = [], []
    for p in psrs:
        t = jnp.asarray(p.toas) - tref
        s.append(jnp.sin(2 * jnp.pi * f0 * t))
        c.append(jnp.cos(2 * jnp.pi * f0 * t))
    return s, c


def precompute_templates(psrs, f0, gnoise, tref):
    """
    :return: dict con S_ss, S_cc, S_sc (n_psr x n_psr), W (n_glob x 2 n_psr)
             = Sigma^-1 V, e i template sin/cos per pulsar.
    """
    n_psr = len(psrs)
    n_glob = gnoise['n_glob']
    s, c = _templates(psrs, f0, tref)

    V = jnp.zeros((n_glob, 2 * n_psr))
    wss, wcc, wsc = [], [], []
    for i, info in enumerate(gnoise['psr']):
        Ni, T, idx = info['Ni'], info['T'], info['idx']
        V = V.at[idx, i].set(T.T @ (Ni * s[i]))
        V = V.at[idx, n_psr + i].set(T.T @ (Ni * c[i]))
        wss.append(jnp.dot(s[i], Ni * s[i]))
        wcc.append(jnp.dot(c[i], Ni * c[i]))
        wsc.append(jnp.dot(s[i], Ni * c[i]))

    W = jsl.lu_solve(gnoise['Sigma_lu'], V)
    G = V.T @ W
    G = 0.5 * (G + G.T)

    S_ss = jnp.diag(jnp.array(wss)) - G[:n_psr, :n_psr]
    S_cc = jnp.diag(jnp.array(wcc)) - G[n_psr:, n_psr:]
    S_sc = jnp.diag(jnp.array(wsc)) - G[:n_psr, n_psr:]

    return {'S_ss': S_ss, 'S_cc': S_cc, 'S_sc': S_sc, 'W': W, 's': s, 'c': c}


# ── Prodotti con i dati ──────────────────────────────────────────────────────

def data_products_batch(residuals_batch, gnoise, tpl):
    """
    :param residuals_batch: lista per pulsar di array (N, ntoa_i)
    :return: R_s, R_c di shape (N, n_psr)
    """
    n_psr = len(residuals_batch)
    N_inj = residuals_batch[0].shape[0]
    u = jnp.zeros((N_inj, gnoise['n_glob']))
    rs, rc = [], []
    for i, info in enumerate(gnoise['psr']):
        Ni, T, idx = info['Ni'], info['T'], info['idx']
        rN = jnp.asarray(residuals_batch[i]) * Ni[None, :]
        u = u.at[:, idx].add(rN @ T)
        rs.append(rN @ tpl['s'][i])
        rc.append(rN @ tpl['c'][i])
    rs = jnp.stack(rs, axis=1)
    rc = jnp.stack(rc, axis=1)
    uW = u @ tpl['W']
    return rs - uW[:, :n_psr], rc - uW[:, n_psr:]


# ── Fe su griglia di cielo ───────────────────────────────────────────────────

def _fe_maps_hd(R_s, R_c, tpl, pos, gw_skyloc):
    S_ss, S_cc, S_sc = tpl['S_ss'], tpl['S_cc'], tpl['S_sc']
    S_cs = S_sc.T
    fpc = jax.vmap(fpc_fast, in_axes=(0, None, None))

    @jax.jit
    def run(R_s, R_c, gw_skyloc):
        def one_sky(gw_pos):
            Fp, Fc, _ = fpc(pos, gw_pos[0], gw_pos[1])
            q = lambda X, S, Y: X @ S @ Y
            M = jnp.array([
                [q(Fp, S_ss, Fp), q(Fp, S_sc, Fp), q(Fp, S_ss, Fc), q(Fp, S_sc, Fc)],
                [q(Fp, S_cs, Fp), q(Fp, S_cc, Fp), q(Fp, S_cs, Fc), q(Fp, S_cc, Fc)],
                [q(Fc, S_ss, Fp), q(Fc, S_sc, Fp), q(Fc, S_ss, Fc), q(Fc, S_sc, Fc)],
                [q(Fc, S_cs, Fp), q(Fc, S_cc, Fp), q(Fc, S_cs, Fc), q(Fc, S_cc, Fc)],
            ])
            Nv = jnp.stack([R_s @ Fp, R_c @ Fp, R_s @ Fc, R_c @ Fc], axis=-1)  # (N, 4)
            x = jnp.linalg.solve(M, Nv.T)                                      # (4, N)
            return 0.5 * jnp.sum(Nv.T * x, axis=0)                             # (N,)
        return jax.vmap(one_sky)(gw_skyloc.T).T                                # (N, n_sky)

    return run(R_s, R_c, gw_skyloc)



# ── Wrapper drop-in: GPU_FeStat ──────────────────────────────────────────────
#
# Stessa interfaccia della versione precedente. Se la likelihood ha un globalgp
# (background CRN) e i params sono dati, usa la Fe con ORF completa nel rumore
# (inner product globale). Altrimenti (background=False) si comporta esattamente
# come prima. Il pulsar term riguarda solo l'iniezione: la Fe resta Earth-term.
#
# Per tornare alla versione a blocchi diagonali senza toccare i chiamanti:
#     export FE_CORRELATED=0
# oppure GPU_FeStat(..., correlated=False).

class GPU_FeStat(GPU_FeStat_BlockDiag):

    def __init__(self, psrs, lik=None, params=None, n_crn=60, tref=None,
                 correlated=None):
        super().__init__(psrs, lik=lik, params=params, n_crn=n_crn, tref=tref)
        if correlated is None:
            correlated = os.environ.get('FE_CORRELATED', '1') != '0'
        has_gp = getattr(lik, 'globalgp', None) is not None
        self.correlated = bool(correlated and has_gp and params is not None)
        self._tpl_hd, self._tpl_hd_f0 = None, None
        if self.correlated:
            print('Fe con ORF completa nel rumore (inner product globale)')
            self._gnoise = build_global_noise(lik, params)
        else:
            print('Fe a blocchi diagonali per pulsar')

    def update_params(self, params):
        super().update_params(params)
        if self.correlated:
            self._gnoise = build_global_noise(self.lik, params)
            self._tpl_hd, self._tpl_hd_f0 = None, None

    def _get_tpl_hd(self, f0):
        if self._tpl_hd is None or self._tpl_hd_f0 != f0:
            self._tpl_hd = precompute_templates(self.psrs, f0, self._gnoise, self.tref)
            self._tpl_hd_f0 = f0
        return self._tpl_hd

    def precompute_M(self, f0):
        if self.correlated:
            return self._get_tpl_hd(f0)
        return super().precompute_M(f0)

    def compute_Fe(self, f0, gw_skyloc, psr_theta_phi=None, M_matrix=None):
        if not self.correlated:
            return super().compute_Fe(f0, gw_skyloc, psr_theta_phi, M_matrix)
        if M_matrix is not None:
            raise ValueError('M_matrix non supportato in modalità correlata.')
        if psr_theta_phi is not None:
            th, ph = psr_theta_phi[:, 0], psr_theta_phi[:, 1]
            pos = jnp.stack([jnp.cos(ph) * jnp.sin(th),
                             jnp.sin(ph) * jnp.sin(th),
                             jnp.cos(th)], axis=1)
        else:
            pos = self.pos
        batch = [jnp.asarray(p.residuals)[None, :] for p in self.psrs]
        tpl = self._get_tpl_hd(f0)
        R_s, R_c = data_products_batch(batch, self._gnoise, tpl)
        return _fe_maps_hd(R_s, R_c, tpl, pos, jnp.asarray(gw_skyloc))[0]

    def compute_Fe_batch(self, f0, gw_skyloc, residuals_batch):
        if not self.correlated:
            return super().compute_Fe_batch(f0, gw_skyloc, residuals_batch)
        tpl = self._get_tpl_hd(f0)
        R_s, R_c = data_products_batch(residuals_batch, self._gnoise, tpl)
        return _fe_maps_hd(R_s, R_c, tpl, self.pos, jnp.asarray(gw_skyloc))
