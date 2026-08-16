"""
Pipeline générique de détection de transit d'exoplanète, pour
n'importe quelle cible Kepler ou TESS.

Installation :
    pip install lightkurve

Usage :
    python3 exoplanet_bls_generic.py "Kepler-69"
    python3 exoplanet_bls_generic.py "TIC 307210830" --mission TESS
    python3 exoplanet_bls_generic.py "TIC 307210830" --mission TESS --period-min 1 --period-max 30
"""

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

# Message cosmétique de lightkurve (log.warning, pas warnings.warn) --
# une estimation qui n'affecte pas le calcul réel quand on fournit
# notre propre grille de périodes (voir exoplanet_bls_count.py).
logging.getLogger("lightkurve").setLevel(logging.ERROR)


def analyser_cible(target, mission="Kepler", period_min=0.5, period_max=20,
                    n_periods=10000, max_sectors=10):
    """Télécharge, nettoie et cherche un transit périodique sur une cible."""

    # exptime fixe la cadence : sans ça, TESS peut renvoyer un mélange
    # de produits 2 minutes ET 20 secondes pour les mêmes secteurs
    # (mission étendue), ce qui double le volume de données pour rien.
    if mission == "Kepler":
        author, exptime = "Kepler", "long"
    else:
        author, exptime = "SPOC", 120

    print(f"Recherche de courbes de lumière pour {target} ({mission})...")
    search_result = lk.search_lightcurve(target, author=author, exptime=exptime)

    if len(search_result) == 0:
        print(f"Aucune donnée trouvée pour '{target}'. "
              f"Vérifie le nom, l'identifiant TIC/KIC, ou la mission.")
        return None

    print(search_result)
    if len(search_result) > max_sectors:
        print(f"({len(search_result)} secteurs disponibles, limité aux {max_sectors} premiers)")
        search_result = search_result[:max_sectors]

    lc_collection = search_result.download_all()

    # Nettoyage : recollage de toutes les observations, aplanissement
    # de la tendance longue, retrait des valeurs aberrantes
    lc = lc_collection.stitch().flatten(window_length=901).remove_outliers()

    # Recherche BLS sur la grille de périodes demandée
    period_grid = np.linspace(period_min, period_max, n_periods)
    bls = lc.to_periodogram(method="bls", period=period_grid, frequency_factor=500)

    best_period = bls.period_at_max_power
    best_t0 = bls.transit_time_at_max_power
    best_duration = bls.duration_at_max_power

    # Indicateur grossier de significativité : le pic doit ressortir
    # nettement au-dessus du niveau de bruit médian du périodogramme.
    # Ce n'est qu'un indice, pas une preuve statistique rigoureuse.
    snr_proxy = float(bls.max_power / np.median(bls.power))

    print(f"\nCible                  : {target}")
    print(f"Période détectée       : {best_period.value:.4f} j")
    print(f"Époque de transit (t0) : {best_t0.value:.4f}")
    print(f"Durée de transit       : {best_duration.value:.4f} j")
    ratio_duree = best_duration.value / best_period.value
    print(f"Ratio durée/période    : {ratio_duree:.1%}"
          f"{'  ⚠ suspect (>15%, rarement un vrai transit)' if ratio_duree > 0.15 else ''}")
    print(f"Puissance / bruit (approx.) : {snr_proxy:.1f}"
          f"  (repère grossier : >20 = signal probable, <5 = suspect)")

    fig, axes = plt.subplots(3, 1, figsize=(9, 11))
    bls.plot(ax=axes[0])
    axes[0].set_title(f"Périodogramme BLS — {target}")

    # Vue du cycle complet (phase normalisée -0.5 à +0.5, indépendante
    # de la durée réelle de la période) : utile pour repérer une
    # éventuelle éclipse secondaire à phase ~0.5, signature typique
    # d'une binaire à éclipses plutôt qu'une planète.
    folded_complet = lc.fold(period=best_period, epoch_time=best_t0, normalize_phase=True)
    folded_complet.scatter(ax=axes[1])
    axes[1].set_title("Cycle complet (phase normalisée) — chercher une 2e éclipse à ±0.5")
    axes[1].set_xlabel("Phase (fraction de période)")

    # Zoom adapté sur le transit lui-même : fenêtre proportionnelle à
    # la durée détectée plutôt qu'une largeur fixe, pour que la forme
    # du creux reste lisible quelle que soit la période.
    folded = lc.fold(period=best_period, epoch_time=best_t0)
    folded.scatter(ax=axes[2])
    demi_fenetre = max(3 * best_duration.value, 0.03 * best_period.value)
    axes[2].set_xlim(-demi_fenetre, demi_fenetre)
    axes[2].set_title(f"Zoom sur le transit — période = {best_period.value:.4f} j")

    plt.tight_layout()
    out_name = f"resultat_{target.replace(' ', '_')}.png"
    plt.savefig(out_name, dpi=150)
    print(f"\nGraphique sauvegardé dans {out_name}")

    return {"period": best_period, "t0": best_t0,
            "duration": best_duration, "snr_proxy": snr_proxy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recherche générique de transit d'exoplanète par BLS")
    parser.add_argument(
        "target",
        help="Nom de cible, ex: 'Kepler-69', 'TIC 307210830', 'TOI-700'")
    parser.add_argument("--mission", default="Kepler", choices=["Kepler", "TESS"])
    parser.add_argument("--period-min", type=float, default=0.5,
                         help="Période minimale à tester, en jours")
    parser.add_argument("--period-max", type=float, default=20,
                         help="Période maximale à tester, en jours")
    args = parser.parse_args()

    analyser_cible(args.target, mission=args.mission,
                    period_min=args.period_min, period_max=args.period_max)
