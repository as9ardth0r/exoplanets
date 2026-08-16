"""
Carte des étoiles observées, en coordonnées galactiques, agrégée sur
TOUS les secteurs scannés (pas un seul) -- avec mise en évidence des
candidats selon leur verdict réel (verdicts.csv), pas juste "a passé
les filtres automatiques".

Installation :
    pip install astropy matplotlib pandas

Usage (lancé depuis exoplanets-site, avec exoplanets cloné à côté) :
    python3 carte_galactique.py --targets "../exoplanets/all_targets_S*_v1.csv" --verdicts verdicts.csv
"""

import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

COULEURS_VERDICT = {
    "confirm": ("#5E9C6B", "Redétections confirmées"),
    "candidat": ("#E2A63B", "Candidats TESS officiels"),
    "marginal": ("#8b8570", "Non tranchés"),
    "reject": ("#B8493D", "Écartés (binaires, variables...)"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="all_targets_S*_v1.csv",
                         help="Motif glob vers les fichiers de cibles de tous les secteurs")
    parser.add_argument("--verdicts", default="verdicts.csv")
    parser.add_argument("--output", default="carte_galactique.png")
    args = parser.parse_args()

    fichiers = sorted(glob.glob(args.targets))
    if not fichiers:
        raise SystemExit(f"Aucun fichier ne correspond à '{args.targets}'")

    targets = pd.concat([pd.read_csv(f, comment="#") for f in fichiers], ignore_index=True)
    targets = targets.drop_duplicates(subset="TICID")  # étoiles vues dans plusieurs secteurs qui se chevauchent
    print(f"{len(fichiers)} secteur(s), {len(targets)} étoile(s) unique(s) au total")

    try:
        verdicts = pd.read_csv(args.verdicts)
        verdicts["TICID"] = verdicts["tic_id"].astype(int)
    except FileNotFoundError:
        verdicts = pd.DataFrame(columns=["TICID", "stamp_class"])
        print(f"{args.verdicts} introuvable -- carte sans mise en évidence de candidats")

    targets = targets.merge(verdicts[["TICID", "stamp_class"]], on="TICID", how="left")
    targets["stamp_class"] = targets["stamp_class"].fillna("aucun")

    coords = SkyCoord(ra=targets["RA"].values * u.deg, dec=targets["Dec"].values * u.deg, frame="icrs")
    gal = coords.galactic
    l_rad = gal.l.wrap_at(180 * u.deg).radian
    b_rad = gal.b.radian

    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(121, projection="mollweide")

    aucun = targets["stamp_class"].values == "aucun"
    sc = ax1.scatter(l_rad[aucun], b_rad[aucun], c=targets.loc[aucun, "Tmag"],
                      cmap="viridis_r", s=8, alpha=0.5, label="Étoiles observées")

    for classe, (couleur, label) in COULEURS_VERDICT.items():
        masque = targets["stamp_class"].values == classe
        if masque.any():
            ax1.scatter(l_rad[masque], b_rad[masque], c=couleur, s=100, marker="*",
                        edgecolor="black", linewidth=0.5,
                        label=f"{label} ({masque.sum()})", zorder=5)

    ax1.set_xticklabels(["150°", "120°", "90°", "60°", "30°", "0°",
                          "-30°", "-60°", "-90°", "-120°", "-150°"])
    ax1.set_title(f"Coordonnées galactiques -- {len(fichiers)} secteurs, {len(targets)} étoiles\n"
                  "(centre = direction du centre de notre galaxie)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=7)
    cbar = plt.colorbar(sc, ax=ax1, orientation="horizontal", pad=0.08, shrink=0.7)
    cbar.set_label("Magnitude TESS (plus petit = plus brillant)")

    ax2 = fig.add_subplot(122)
    ax2.set_facecolor("black")
    rng = np.random.default_rng(0)
    for bras in range(4):
        theta = np.linspace(0, 3 * np.pi, 300) + bras * np.pi / 2
        r = 0.4 * np.exp(0.17 * theta)
        r = r / r.max() * 8.5
        theta_jitter = theta + rng.normal(0, 0.05, size=theta.shape)
        r_jitter = r * (1 + rng.normal(0, 0.05, size=r.shape))
        x, y = r_jitter * np.cos(theta_jitter), r_jitter * np.sin(theta_jitter)
        ax2.scatter(x, y, s=1, c="lightblue", alpha=0.4)

    ax2.scatter(0, 0, s=300, c="gold", marker="*", zorder=5, label="Centre galactique")
    soleil_x, soleil_y = 8.0, 0
    ax2.scatter(soleil_x, soleil_y, s=60, c="white", edgecolor="orange",
                linewidth=1.5, zorder=6, label="Le Soleil")
    ax2.scatter(soleil_x, soleil_y, s=25, c="red", marker="*", zorder=7,
                label="Étoiles scannées\n(toutes ici, invisibles à cette échelle)")

    ax2.set_xlim(-12, 12)
    ax2.set_ylim(-12, 12)
    ax2.set_aspect("equal")
    ax2.set_xlabel("kpc")
    ax2.set_ylabel("kpc")
    ax2.set_title(f"Échelle : {len(fichiers)} secteurs TESS dans la Voie lactée\n"
                  "(vue schématique, pas un vrai modèle galactique)", color="black")
    ax2.legend(loc="upper right", fontsize=8, facecolor="white", framealpha=0.9)
    ax2.tick_params(colors="black")
    for spine in ax2.spines.values():
        spine.set_color("black")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Carte sauvegardée dans {args.output}")


if __name__ == "__main__":
    main()
