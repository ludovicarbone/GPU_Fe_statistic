import jax
import jax.numpy as jnp
import numpy as np
import discovery as ds
from discovery import const as const #ma non c'è un corrispettivo discovery??


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


def makemodel_cgw(psrs, cwcommon, pulsterm=False):
    cgw_delay = makedelay_binary(pulsterm=pulsterm)

    pslmodels = []
    tspan = ds.getspan(psrs)

    for p in psrs:
        model = [p.residuals,
                 ds.makenoise_measurement(p, p.noisedict, tnequad=True),
                 ds.makegp_timing(p, svd=True, variance=1e-40),
                 ds.makedelay(p, cgw_delay, name='cw', common=cwcommon)]
        '''
        if p.noisedict.get(p.name + '_dm_gp_components', 0):
            model.append(ds.makegp_fourier(p, ds.powerlaw, p.noisedict[p.name + '_dm_gp_components'],
                                           T=ds.getspan(p), name='dm_gp',
                                           fourierbasis=ds.make_dmfourierbasis(alpha=2.0, tndm=True)))

        if p.noisedict.get(p.name + '_red_components', 0):
            model.append(ds.makegp_fourier(p, ds.powerlaw, p.noisedict[p.name + '_red_components'],
                                           T=tspan, name='red_noise'))
        '''

        pslmodels.append(ds.PulsarLikelihood(model))

    return ds.GlobalLikelihood(pslmodels)


def run_injection(d_psrs, cwpars, cwcommon, pulsterm=False, key=None):
    if key is None:
        key = jax.random.key(345)

    """
    Parameters
    ----------
    d_psrs   : lista di pulsar
    cwpars   : lista [sindec, cosinc, log10_f0, log10_h0, phi_earth, psi, ra, log10_Mc]
    cwcommon : lista dei nomi dei parametri CW comuni
    pulsterm : bool, se True inietta anche il pulsar term
    seed     : int, seed JAX per la simulazione

    Returns
    -------
    residuals : array dei residui simulati
    model     : GlobalLikelihood usato per la simulazione
    """
    model = makemodel_cgw(d_psrs, cwcommon, pulsterm=pulsterm)

    cgw_params_inj = {}
    for psr in d_psrs:
        cgw_params_inj.update(psr.noisedict) 
        cgw_params_inj[psr.name + '_cw_d_psr'] = 1.0
    for name, par in zip(cwcommon, cwpars):
        cgw_params_inj[name] = par

    _, residuals = model.sample(key, cgw_params_inj)

    return residuals, model

