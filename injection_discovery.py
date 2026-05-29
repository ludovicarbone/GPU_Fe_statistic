import jax
import jax.numpy as jnp
import numpy as np
import discovery as ds
from discovery import const as const


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

    return fplus, fcross, omhat_dot_pos


def makedelay_binary(evolve=True, pulsterm=False):
    def delay_binary(toas, pos, pdist, log10_h0, log10_f0, ra, sindec, cosinc, psi, phi_earth, d_psr, log10_Mc):

        # --- Precompute parameters ---
        h0 = 10.0**log10_h0
        f0 = 10.0**log10_f0
        dec, inc = jnp.arcsin(sindec), jnp.arccos(cosinc)

        # Antenna pattern
        fplus, fcross, cosMu = fpc_fast(pos, 0.5 * jnp.pi - dec, ra)

        # Mass and distance
        mc    = 10.0**log10_Mc * const.Tsun
        mc53  = mc**(5.0/3.0)
        w0    = jnp.pi * f0
        w0_83 = w0**(8.0/3.0)
        w0_m53= w0**(-5.0/3.0)
        dist  = 2.0 * mc53 * (jnp.pi * f0)**(2.0/3.0) / h0
        phi_earth = phi_earth / 2.0

        # --- Frequency evolution: Earth term ---
        coef  = (256.0/5.0) * mc53 * w0_83
        omega = w0 * (1.0 - coef * toas)**(-3.0/8.0)
        phase = phi_earth + (1.0/32.0) / mc53 * (w0_m53 - omega**(-5.0/3.0))

        # --- Waveform amplitudes: Earth term ---
        At    = -0.5 * jnp.sin(2*phase) * (3.0 + jnp.cos(2*inc))
        Bt    =  2.0 * jnp.cos(2*phase) * jnp.cos(inc)
        alpha = mc53 / (dist * omega**(1.0/3.0))

        rplus  = alpha * (-At * jnp.cos(2*psi) + Bt * jnp.sin(2*psi))
        rcross = alpha * ( At * jnp.sin(2*psi) + Bt * jnp.cos(2*psi))

        if pulsterm:
            # --- Pulsar distance and retarded time ---
            p_dist  = (pdist[0] + pdist[1] * d_psr) * const.kpc / const.c
            tp      = toas - p_dist * (1.0 - cosMu)

            # --- Frequency evolution: Pulsar term ---
            omega_p = w0 * (1.0 - coef * tp)**(-3.0/8.0)
            phase_p = phi_earth + (1.0/32.0) / mc53 * (w0_m53 - omega_p**(-5.0/3.0))

            # --- Waveform amplitudes: Pulsar term ---
            At_p    = -0.5 * jnp.sin(2*phase_p) * (3.0 + jnp.cos(2*inc))
            Bt_p    =  2.0 * jnp.cos(2*phase_p) * jnp.cos(inc)
            alpha_p = mc53 / (dist * omega_p**(1.0/3.0))

            rplus_p  = alpha_p * (-At_p * jnp.cos(2*psi) + Bt_p * jnp.sin(2*psi))
            rcross_p = alpha_p * ( At_p * jnp.sin(2*psi) + Bt_p * jnp.cos(2*psi))

            # Earth + Pulsar term
            res = fplus * (rplus_p - rplus) + fcross * (rcross_p - rcross)
        else:
            # Solo Earth term
            res = fplus * (-rplus) + fcross * (-rcross)

        return res

    return delay_binary


# ── CHANGED: added background and orf parameters ─────────────────────────────
def makemodel_cgw(psrs, cwcommon, orf=ds.hd_orf, pulsterm=False, background=False):
    """
    Build a GlobalLikelihood for CW injection / recovery.

    Parameters
    ----------
    psrs       : list of pulsars
    cwcommon   : list of common CW parameter names
    orf        : overlap reduction function (default Hellings-Downs)
    pulsterm   : bool, include pulsar term in the CW waveform
    background : bool, if True add a correlated-red-noise global GP
    """
    cgw_delay = makedelay_binary(pulsterm=pulsterm)

    pslmodels = []
    tspan = ds.getspan(psrs)

    for p in psrs:
        model = [p.residuals,
                 ds.makenoise_measurement(p, p.noisedict, tnequad=True),
                 ds.makegp_timing(p, svd=True, variance=1e-40),
                 ds.makedelay(p, cgw_delay, name='cw', common=cwcommon)]

        if p.noisedict.get(p.name + '_dm_gp_components', 0):
            model.append(ds.makegp_fourier(p, ds.powerlaw, p.noisedict[p.name + '_dm_gp_components'],
                                           T=ds.getspan(p), name='dm_gp',
                                           fourierbasis=ds.make_dmfourierbasis(alpha=2.0, tndm=True)))

        if p.noisedict.get(p.name + '_red_components', 0):
            model.append(ds.makegp_fourier(p, ds.powerlaw, p.noisedict[p.name + '_red_components'],
                                           T=tspan, name='red_noise'))

        pslmodels.append(ds.PulsarLikelihood(model))

    if background:
        return ds.GlobalLikelihood(
            psls=pslmodels,
            globalgp=ds.makegp_fourier_global(psrs, ds.powerlaw, orf,
                                              components=30, T=tspan, name='crn'),
        )
    else:
        return ds.GlobalLikelihood(pslmodels)


# ── CHANGED: added background, orf, bg_params to run_injection ───────────────
def run_injection(d_psrs, cwpars, cwcommon, pulsterm=False,
                  background=False, orf=ds.hd_orf, bg_params=None, key=None):
    """
    Simulate a CW injection (optionally on top of a correlated background).

    Parameters
    ----------
    d_psrs     : list of pulsars
    cwpars     : list of CW parameter values matching cwcommon
    cwcommon   : list of CW common-parameter names (NO background params here)
    pulsterm   : bool, include pulsar term
    background : bool, if True inject a CRN background as well
    orf        : overlap reduction function (used only when background=True)
    bg_params  : dict with background parameters, e.g.
                 {'crn_log10_A': -15.0, 'crn_gamma': 4.33}
                 Required when background=True.
    key        : JAX random key

    Returns
    -------
    residuals : simulated residuals
    model     : GlobalLikelihood used for the simulation
    """
    if key is None:
        key = jax.random.key(345)

    model = makemodel_cgw(d_psrs, cwcommon, orf=orf,
                          pulsterm=pulsterm, background=background)

    # Build the full parameter dictionary
    cgw_params_inj = {}
    for psr in d_psrs:
        cgw_params_inj.update(psr.noisedict)
        cgw_params_inj[psr.name + '_cw_d_psr'] = 1.0
    for name, par in zip(cwcommon, cwpars):
        cgw_params_inj[name] = par

    # Add background parameters when needed
    if background:
        if bg_params is None:
            bg_params = {'crn_log10_A': -15.0, 'crn_gamma': 4.33}
        cgw_params_inj.update(bg_params)

    _, residuals = model.sample(key, cgw_params_inj)

    return residuals, model