"""
Triage rapide de plusieurs candidats en une seule image : un panneau
"cycle complet" par étoile, en grille. Objectif : repérer d'un coup
d'œil les cas évidents (onde continue = variable stellaire, double
éclipse = binaire) sans avoir à lancer exoplanet_bls_generic.py une
étoile à la fois. Une fois les cas prometteurs identifiés ici, ne fais
l'analyse complète (exoplanet_bls_generic.py, 3 panneaux) que sur ceux-là.

Installation :
    pip install lightkurve matplotlib numpy

Usage :
    python3 triage_rapide.py 110340485 89045042 146536983 64053673 33742428 114018671 54002166 630017074
"""

import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

logging.getLogger("lightkurve").setLevel(logging.ERROR)


def analyser_rapide(tic_id, period_min=0.5, period_max=15, n_periods=5000, max_sectors=10):
    """Version allégée : une seule recherche BLS, pas de masquage
    multi-planète, juste de quoi tracer le cycle complet."""
    search_result = lk.search_lightcurve(f"TIC {tic_id}", author="SPOC", exptime=120)
    if len(search_result) == 0:
        return None
    if len(search_result) > max_sectors:
        search_result = search_result[:max_sectors]

    lc = search_result.download_all().stitch().flatten(window_length=901).remove_outliers()

    period_grid = np.linspace(period_min, period_max, n_periods)
    bls = lc.to_periodogram(method="bls", period=period_grid, frequency_factor=2000)

    best_period = bls.period_at_max_power
    best_t0 = bls.transit_time_at_max_power
    snr_proxy = float(bls.max_power / np.median(bls.power))

    folded = lc.fold(period=best_period, epoch_time=best_t0, normalize_phase=True)
    return folded, best_period.value, snr_proxy


if __name__ == "__main__":
    tic_ids = sys.argv[1:]
    if not tic_ids:
        print("Usage : python3 triage_rapide.py <tic_id_1> <tic_id_2> ...")
        sys.exit(1)

    n = len(tic_ids)
    cols = 3
    rows = -(-n // cols)  # arrondi au-dessus

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i, tic_id in enumerate(tic_ids):
        ax = axes[i]
        print(f"TIC {tic_id}...", end=" ", flush=True)
        try:
            resultat = analyser_rapide(tic_id)
            if resultat is None:
                ax.set_title(f"TIC {tic_id}\n(aucune donnée)")
                print("aucune donnée")
                continue
            folded, period, snr = resultat
            ax.scatter(folded.phase.value, folded.flux.value, s=1, alpha=0.4)
            ax.set_title(f"TIC {tic_id}\npériode={period:.4f}j, snr≈{snr:.1f}", fontsize=10)
            ax.set_xlabel("Phase")
            print(f"OK (période={period:.4f}, snr={snr:.1f})")
        except Exception as e:
            ax.set_title(f"TIC {tic_id}\n(erreur)")
            print(f"erreur: {e}")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("triage_rapide.png", dpi=150)
    print("\nGrille sauvegardée dans triage_rapide.png")
