"""
Costruzione generalizzata di (T, Sigma, Ni_diag) per pulsar, per la Fe-statistic.

Sostituisce build_per_pulsar_sigma e build_per_pulsar_white_noise di
GPU_Fe_statistics.py con un'unica funzione che gestisce, in qualunque
combinazione:

  - rumore bianco (efac / equad)
  - timing model            -> marginalizzato (prior piatto)
  - GP per pulsar variabili -> rumore rosso, DM
  - CRN globale             -> approssimazione a blocchi diagonali, con il
                               prior MARGINALE di ciascun pulsar (Phi_ii)^-1

La forma dell'inner product resta quella di inner_product_jax:

    (x|y) = x^T N^-1 y - (T^T N^-1 x)^T Sigma^-1 (T^T N^-1 y)
    Sigma = T^T N^-1 T + Phi^-1

quindi basta costruire T e Phi^-1 con i blocchi giusti.

--------------------------------------------------------------------------
NOTA sulla convenzione del prior (verificata su discovery/matrix.py)
--------------------------------------------------------------------------
NoiseMatrix1D_novar.N e la matrice Phi, NON la sua inversa: WoodburyKernel_novar
la inverte con P.inv() -> diag(1/N). Il codice precedente usava
Phi_inv_tm = diag(psl.N.P.N), cioe Phi al posto di Phi^-1. Con
makegp_timing(variance=1e-40) si ottiene Phi ~ 1e-38, trascurabile accanto a
FtNmF, quindi il prior risultava piatto e il timing veniva marginalizzato:
il comportamento voluto, ma ottenuto per compensazione fra due errori.

Qui il blocco del timing viene messo a ZERO esplicitamente (prior improprio
piatto = marginalizzazione completa), che e cio che la Fe-statistic richiede
e non dipende piu dal valore di `variance` usato nel modello di iniezione.
"""

import numpy as np
import jax.numpy as jnp


# ── Smontaggio del kernel di un singolo pulsar ───────────────────────────────

def unpack_pulsar_kernel(psl):
    """
    Estrae le componenti di rumore da una PulsarLikelihood, indipendentemente
    dal fatto che contenga o meno GP a parametri liberi.

    Due strutture possibili (verificate per introspezione):

      senza GP variabili — WoodburyKernel_novar
        psl.N.F       (ntoa, n_tm)   base del timing
        psl.N.P.N     (n_tm,)        Phi del timing
        psl.N.N.N     (ntoa,)        diagonale del rumore bianco

      con GP variabili — WoodburyKernel_varP
        psl.N.F       (ntoa, n_gp)   base dei soli GP variabili
        psl.N.P_var                  prior dipendente dai parametri
        psl.N.index                  {nome: slice} dei blocchi dentro F
        psl.N.N                      WoodburyKernel_novar annidato, che
                                     contiene bianco + timing come sopra

    :param psl: PulsarLikelihood
    :return:    dict con chiavi Ni_diag, F_tm, P_tm, F_gp, P_var, index
    """
    K = psl.N

    if hasattr(K, 'P_var') and getattr(K, 'P_var', None) is not None:
        inner = K.N                      # WoodburyKernel_novar: bianco + timing
        F_gp = np.asarray(K.F)
        P_var = K.P_var
        index = getattr(K, 'index', None)
    else:
        inner = K
        F_gp = None
        P_var = None
        index = None

    if not hasattr(inner, 'N') or not hasattr(inner.N, 'N'):
        raise TypeError(
            f'Kernel non supportato: {type(inner).__name__}. build_per_pulsar_noise '
            'gestisce WoodburyKernel_novar e WoodburyKernel_varP, cioe rumore bianco '
            'costante (makenoise_measurement con noisedict fissato). Con efac/equad '
            'a parametri liberi (WoodburyKernel_varN, _varNP, _varFP) N dipende dai '
            'parametri: Ni_diag andrebbe ricostruito a ogni chiamata dai params, '
            'non letto una volta sola da inner.N.N.')

    F_tm = np.asarray(inner.F)
    P_tm = np.asarray(inner.P.N)
    Ni_diag = 1.0 / jnp.asarray(inner.N.N)

    return {'Ni_diag': Ni_diag, 'F_tm': F_tm, 'P_tm': P_tm, 'F_gp': F_gp,
            'P_var': P_var, 'index': index}


# ── Offset dei blocchi CRN dentro il Phi^-1 globale ──────────────────────────

def crn_block_slices(globalgp):
    """
    Slice del blocco CRN di ciascun pulsar dentro la matrice Phi^-1 globale.

    Il codice precedente assumeva n_crn uguale per tutti e usava
    Pinv[i*n_crn:(i+1)*n_crn]. Qui gli offset sono cumulati dalle shape reali,
    cosi resta valido anche se il numero di componenti varia per pulsar.

    :param globalgp: lik.globalgp
    :return:         lista di slice, una per pulsar
    """
    slices, off = [], 0
    for F in globalgp.Fs:
        n = np.asarray(F).shape[1]
        slices.append(slice(off, off + n))
        off += n
    return slices


# ── Builder unificato ────────────────────────────────────────────────────────

def build_per_pulsar_noise(lik, params=None, marginalize_timing=True,
                           verbose=True):
    """
    Costruisce (T, Sigma, Ni_diag) per ogni pulsar, includendo tutte le
    componenti gaussiane presenti nella likelihood.

    T     = [ F_timing | F_gp_pulsar | F_crn ]   (le parti presenti)
    Phi^-1 = blocchi diagonali corrispondenti:
             timing  -> 0            (prior piatto, marginalizzazione)
             GP psr  -> P_var.make_inv()(params)
             CRN     -> (Phi_ii)^-1, inversa del blocco i-esimo della Phi
                        globale. NON il blocco i-esimo di Phi^-1: quello e la
                        precisione del CRN nel pulsar i condizionata al CRN
                        negli altri pulsar, (Phi^-1)_ii = (Gamma^-1)_ii phi^-1,
                        e sottostima il CRN di un fattore (Gamma^-1)_ii >= 1
                        (1.15-1.66 per EPTA DR2new). Poiche la Fe tratta i
                        pulsar separatamente, il prior coerente e il marginale.

    :param lik:    GlobalLikelihood di discovery
    :param params: dict con i parametri di rumore (CRN e/o RN/DM per pulsar).
                   Obbligatorio se la likelihood contiene un CRN globale o GP
                   a parametri liberi: in quel caso, se manca, viene sollevata
                   un'eccezione invece di scartare silenziosamente il blocco.
    :param marginalize_timing: se True il blocco timing ha Phi^-1 = 0.
                   Metterlo a False usa il prior effettivo del modello, che con
                   variance=1e-40 inchioda il timing invece di marginalizzarlo:
                   quasi certamente NON e cio che vuoi per la Fe-statistic.
    :param verbose: stampa un riepilogo della composizione di T
    :return:       lista di (T, Sigma, Ni_diag), una tupla per pulsar
    """
    globalgp = getattr(lik, 'globalgp', None)

    if globalgp is not None and params is None:
        raise ValueError(
            'La likelihood contiene un CRN globale ma params e None: servono '
            'crn_log10_A e crn_gamma. Senza, il CRN verrebbe scartato in '
            'silenzio e la Fe sarebbe calcolata come se il rumore fosse solo '
            'bianco, su residui che invece contengono il background.')

    has_crn = globalgp is not None

    if has_crn:
        P_glob_inv = globalgp.Phi_inv or globalgp.Phi.make_inv()
        Pinv_crn, _ = P_glob_inv(params)
        crn_slices = crn_block_slices(globalgp)
        # covarianza globale: serve il blocco marginale (Phi_ii)^-1, non (Phi^-1)_ii
        Phi_crn = np.linalg.inv(np.asarray(Pinv_crn))

    results = []
    for i, psl in enumerate(lik.psls):
        parts = unpack_pulsar_kernel(psl)
        Ni_diag = parts['Ni_diag']

        blocks_F = [jnp.asarray(parts['F_tm'])]
        blocks_Pinv = []
        labels = [f"timing({parts['F_tm'].shape[1]})"]

        n_tm = parts['F_tm'].shape[1]
        if marginalize_timing:
            blocks_Pinv.append(jnp.zeros((n_tm, n_tm)))
        else:
            # prior effettivo del modello: Phi^-1 = 1 / Phi.
            # Va preso dal kernel *interno* (parts['P_tm']): con GP variabili
            # psl.N e un WoodburyKernel_varP che non espone .P.
            blocks_Pinv.append(jnp.diag(1.0 / jnp.asarray(parts['P_tm'])))

        # GP per pulsar (rumore rosso, DM, ...)
        if parts['P_var'] is not None:
            if params is None:
                raise ValueError(
                    'La likelihood contiene GP a parametri liberi ma params e '
                    'None: servono i valori di log10_A / gamma per pulsar.')
            missing = [p for p in parts['P_var'].params if p not in params]
            if missing:
                raise KeyError(f'Parametri mancanti in params: {missing}')

            Pinv_gp, _ = parts['P_var'].make_inv()(params)
            blocks_F.append(jnp.asarray(parts['F_gp']))
            blocks_Pinv.append(jnp.asarray(Pinv_gp))
            if parts['index']:
                labels += [k for k in parts['index']]
            else:
                labels.append(f"gp({parts['F_gp'].shape[1]})")

        # CRN globale
        if has_crn:
            F_crn = jnp.asarray(globalgp.Fs[i])
            sl = crn_slices[i]
            blocks_F.append(F_crn)
            blocks_Pinv.append(jnp.asarray(np.linalg.inv(Phi_crn[sl, sl])))
            labels.append(f'crn({F_crn.shape[1]})')

        T_tot = jnp.concatenate(blocks_F, axis=1)

        # Phi^-1 diagonale a blocchi
        sizes = [b.shape[0] for b in blocks_Pinv]
        n_tot = int(sum(sizes))
        Phi_inv = jnp.zeros((n_tot, n_tot))
        off = 0
        for b, n in zip(blocks_Pinv, sizes):
            Phi_inv = Phi_inv.at[off:off + n, off:off + n].set(b)
            off += n

        FtNmF = jnp.dot(T_tot.T * Ni_diag[None, :], T_tot)
        Sigma = FtNmF + Phi_inv

        if verbose and i == 0:
            print(f'  composizione di T (pulsar 0): {" + ".join(labels)}'
                  f'  ->  {T_tot.shape[1]} colonne')

        results.append((T_tot, Sigma, Ni_diag))

    if verbose:
        print(f'  costruite {len(results)} matrici Sigma'
              f'{" (con CRN)" if has_crn else " (senza CRN)"}')

    return results
