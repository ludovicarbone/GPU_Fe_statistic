import jax
import jax.numpy as jnp
import numpy as np
import discovery as ds
from discovery import const as const


# ── Reference epoch helper ────────────────────────────────────────────────────
def compute_tref(psrs):
    """
    Data-centered reference epoch (seconds): mean over all TOAs of all pulsars.
    Replaces the old hard-coded 53000 * 86400.

    MUST be used identically by injection, SNR calibration and Fe recovery.
    """
    return float(np.mean(np.concatenate([np.asarray(p.toas) for p in psrs])))


def fpc_fast(pos, gwtheta, gwphi):
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

    return fplus, fcross, omhat_dot_pos   # cosMu = omhat_dot_pos


def makedelay_binary(evolve=True, pulsterm=False, tref=0.0):
    """
    CW Earth (+ optional pulsar) term delay.

    Parameters
    ----------
    pulsterm : bool
        If True include the pulsar term (full independent frequency evolution
        with phi_psr and the retarded reference omega_p0). Earth-only otherwise.
    tref : float
        Reference epoch in seconds. MUST be data-centered (compute_tref) and
        identical to the Fe-recovery tref. Leaving it at 0 evaluates the chirp
        at absolute epoch (~5e9 s) and rescales the observed frequency by
        (1 - coef*tref)^(-3/8) ~ 1.25x  ->  the bin-15 -> bin-19 bug.

    NOTE: phi_psr is an OPTIONAL trailing argument (default 0.0) so that all
    existing positional calls with the old 11-argument signature keep working;
    it is only used when pulsterm=True.
    """
    def delay_binary(toas, pos, pdist,
                     log10_h0, log10_f0, ra, sindec, cosinc, psi,
                     phi_earth, d_psr, log10_Mc, phi_psr=0.0):

        # Reference the chirp to the data epoch (THE FIX).
        t = toas - tref

        h0 = 10.0**log10_h0
        f0 = 10.0**log10_f0
        dec, inc = jnp.arcsin(sindec), jnp.arccos(cosinc)

        # Antenna pattern; cosMu = omhat_dot_pos
        fplus, fcross, cosMu = fpc_fast(pos, 0.5 * jnp.pi - dec, ra)

        # Mass / frequency / distance
        mc     = 10.0**log10_Mc * const.Tsun
        mc53   = mc**(5.0/3.0)
        w0     = jnp.pi * f0
        w0_83  = w0**(8.0/3.0)
        w0_m53 = w0**(-5.0/3.0)
        dist   = 2.0 * mc53 * (jnp.pi * f0)**(2.0/3.0) / h0
        phi_earth = phi_earth / 2.0

        coef  = (256.0/5.0) * mc53 * w0_83

        # ── Earth term ────────────────────────────────────────────────────────
        omega = w0 * (1.0 - coef * t)**(-3.0/8.0)
        phase = phi_earth + (1.0/32.0) / mc53 * (w0_m53 - omega**(-5.0/3.0))

        At    = -0.5 * jnp.sin(2*phase) * (3.0 + jnp.cos(2*inc))
        Bt    =  2.0 * jnp.cos(2*phase) * jnp.cos(inc)
        alpha = mc53 / (dist * omega**(1.0/3.0))

        rplus  = alpha * (-At * jnp.cos(2*psi) + Bt * jnp.sin(2*psi))
        rcross = alpha * ( At * jnp.sin(2*psi) + Bt * jnp.cos(2*psi))

        if pulsterm:
            # Pulsar distance and retarded time (referenced to tref via t)
            p_dist  = (pdist[0] + pdist[1] * d_psr) * const.kpc / const.c
            tp      = t - p_dist * (1.0 - cosMu)

            # Pulsar-term frequency evolution with its own retarded reference
            omega_p  = w0 * (1.0 - coef * tp)**(-3.0/8.0)
            omega_p0 = w0 * (1.0 + coef * p_dist * (1.0 - cosMu))**(-3.0/8.0)

            phase_p = (phi_earth + phi_psr
                       + (1.0/32.0) / mc53
                       * (w0**(-5.0/3.0) - omega_p**(-5.0/3.0)))

            At_p    = -0.5 * jnp.sin(2*phase_p) * (3.0 + jnp.cos(2*inc))
            Bt_p    =  2.0 * jnp.cos(2*phase_p) * jnp.cos(inc)
            alpha_p = mc53 / (dist * omega_p**(1.0/3.0))

            rplus_p  = alpha_p * (-At_p * jnp.cos(2*psi) + Bt_p * jnp.sin(2*psi))
            rcross_p = alpha_p * ( At_p * jnp.sin(2*psi) + Bt_p * jnp.cos(2*psi))

            res = fplus * (rplus_p - rplus) + fcross * (rcross_p - rcross)
        else:
            # Earth term only (phi_psr unused)
            res = fplus * (-rplus) + fcross * (-rcross)

        return res

    return delay_binary


def makemodel_cgw(psrs, cwcommon, orf=ds.hd_orf, pulsterm=False,
                  background=False, tref=None):
    """
    Build a GlobalLikelihood for CW injection / recovery.

    tref : reference epoch (s). If None it is computed data-centered from psrs.
           Pass the SAME value used by the Fe recovery and the SNR calibration.
    """
    if tref is None:
        tref = compute_tref(psrs)

    cgw_delay = makedelay_binary(pulsterm=pulsterm, tref=tref)

    pslmodels = []
    tspan = ds.getspan(psrs)
    for p in psrs:
        model = [p.residuals,
                 ds.makenoise_measurement(p, p.noisedict, tnequad=True),
                 ds.makegp_timing(p, svd=True, variance=1e-40),
                 ds.makedelay(p, cgw_delay, name='cw', common=cwcommon)]
        pslmodels.append(ds.PulsarLikelihood(model))

    if background:
        return ds.GlobalLikelihood(
            psls=pslmodels,
            globalgp=ds.makegp_fourier_global(psrs, ds.powerlaw, orf,
                                              components=30, T=tspan, name='crn'),
        )
    else:
        return ds.GlobalLikelihood(pslmodels)


def run_injection(d_psrs, cwpars, cwcommon, pulsterm=False,
                  background=False, orf=ds.hd_orf, bg_params=None,
                  key=None, tref=None):
    """
    Simulate a CW injection (optionally on top of a correlated background).

    tref : reference epoch (s). If None it is computed data-centered from
           d_psrs and threaded into the model. Pass the SAME value used by
           the Fe recovery / SNR calibration.
    """
    if key is None:
        key = jax.random.key(345)
    if tref is None:
        tref = compute_tref(d_psrs)

    model = makemodel_cgw(d_psrs, cwcommon, orf=orf,
                          pulsterm=pulsterm, background=background, tref=tref)

    cgw_params_inj = {}
    for psr in d_psrs:
        cgw_params_inj.update(psr.noisedict)
        cgw_params_inj[psr.name + '_cw_d_psr']   = 1.0
        cgw_params_inj[psr.name + '_cw_phi_psr'] = 0.0   # phi_psr per-pulsar = 0
    for name, par in zip(cwcommon, cwpars):
        cgw_params_inj[name] = par

    if background:
        if bg_params is None:
            bg_params = {'crn_log10_A': -15.0, 'crn_gamma': 4.33}
        cgw_params_inj.update(bg_params)

    _, residuals = model.sample(key, cgw_params_inj)

    return residuals, model