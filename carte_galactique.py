"""
Carte des étoiles observées (secteur TESS), en coordonnées galactiques
-- centrée sur la direction du centre de notre galaxie -- avec mise en
évidence des candidats détectés par le pipeline BLS. Un second panneau
schématique donne l'échelle : où ce petit échantillon se situe par
rapport à la Voie lactée entière.

Installation :
    pip install astropy matplotlib pandas

Usage :
    python3 carte_galactique.py all_targets_S108_v1.csv resultats_candidats.csv
    python3 carte_galactique.py all_targets_S108_v1.csv   (sans résultats, juste la liste de cibles)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

targets_file = sys.argv[1] if len(sys.argv) > 1 else "all_targets_S108_v1.csv"
resultats_file = sys.argv[2] if len(sys.argv) > 2 else "resultats_candidats.csv"

targets = pd.read_csv(targets_file, comment="#")

# Charger les candidats si le fichier de résultats est disponible.
# Les 5 filtres anti-faux-positifs (bord de grille, ratio durée/période,
# période dupliquée, profondeur excessive, cluster 13.7j) sont déjà
# appliqués directement dans exoplanet_bls_count.py -- n_candidats > 0
# ici veut donc déjà dire "a passé tous les filtres", pas besoin de
# refiltrer une deuxième fois ici.
candidats_ids = set()
try:
    resultats = pd.read_csv(resultats_file)
    resultats["TICID"] = resultats["nom"].str.replace("TIC ", "", regex=False).astype(int)
    candidats_ids = set(resultats[resultats["n_candidats"] > 0]["TICID"])
    print(f"{len(candidats_ids)} candidat(s) (déjà filtré par exoplanet_bls_count.py)")
    # On limite la carte aux étoiles déjà traitées, pour ne pas laisser
    # croire que le reste du lot est sans intérêt -- c'est juste pas encore fait
    traitees_ids = set(resultats["TICID"])
    targets = targets[targets["TICID"].isin(traitees_ids)]
    print(f"Carte limitée aux {len(targets)} étoile(s) déjà traitée(s)")
except FileNotFoundError:
    print(f"{resultats_file} introuvable -- carte de la liste de cibles complète, sans candidats")

targets["est_candidat"] = targets["TICID"].isin(candidats_ids)

# Conversion RA/Dec (équatorial, ICRS) -> coordonnées galactiques (l, b)
coords = SkyCoord(ra=targets["RA"].values * u.deg, dec=targets["Dec"].values * u.deg, frame="icrs")
gal = coords.galactic
l_rad = gal.l.wrap_at(180 * u.deg).radian  # pour la projection Mollweide (-pi à +pi)
b_rad = gal.b.radian

fig = plt.figure(figsize=(14, 8))

# --- Panneau 1 : carte du ciel en coordonnées galactiques ---------------
ax1 = fig.add_subplot(121, projection="mollweide")

autres = ~targets["est_candidat"].values
sc = ax1.scatter(l_rad[autres], b_rad[autres], c=targets.loc[autres, "Tmag"],
                  cmap="viridis_r", s=10, alpha=0.6, label="Étoiles observées")

cand = targets["est_candidat"].values
if cand.any():
    ax1.scatter(l_rad[cand], b_rad[cand], c="red", s=120, marker="*",
                edgecolor="black", linewidth=0.6,
                label=f"Candidats ({cand.sum()})", zorder=5)

ax1.set_xticklabels(["150°", "120°", "90°", "60°", "30°", "0°",
                      "-30°", "-60°", "-90°", "-120°", "-150°"])
ax1.set_title("Coordonnées galactiques\n(centre = direction du centre de notre galaxie)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="lower right", fontsize=8)
cbar = plt.colorbar(sc, ax=ax1, orientation="horizontal", pad=0.08, shrink=0.7)
cbar.set_label("Magnitude TESS (plus petit = plus brillant)")

# --- Panneau 2 : schéma d'échelle -- où ça se situe dans la galaxie -----
ax2 = fig.add_subplot(122)
ax2.set_facecolor("black")

# Galaxie schématique : un bulbe central + bras spiraux logarithmiques
# (illustratif, pas un vrai modèle -- juste pour donner l'échelle)
rng = np.random.default_rng(0)
for bras in range(4):
    theta = np.linspace(0, 3 * np.pi, 300) + bras * np.pi / 2
    r = 0.4 * np.exp(0.17 * theta)
    r = r / r.max() * 8.5  # normalisé à ~8.5 kpc (rayon galactique approx.)
    theta_jitter = theta + rng.normal(0, 0.05, size=theta.shape)
    r_jitter = r * (1 + rng.normal(0, 0.05, size=r.shape))
    x, y = r_jitter * np.cos(theta_jitter), r_jitter * np.sin(theta_jitter)
    ax2.scatter(x, y, s=1, c="lightblue", alpha=0.4)

ax2.scatter(0, 0, s=300, c="gold", marker="*", zorder=5, label="Centre galactique")

# Position approximative du Soleil (~8 kpc du centre)
soleil_x, soleil_y = 8.0, 0
ax2.scatter(soleil_x, soleil_y, s=60, c="white", edgecolor="orange",
            linewidth=1.5, zorder=6, label="Le Soleil")

# Les cibles TESS de ce lot : toutes à quelques centaines de parsecs
# du Soleil au maximum -- un point unique à cette échelle (kpc)
ax2.scatter(soleil_x, soleil_y, s=25, c="red", marker="*", zorder=7,
            label="Étoiles de ce lot\n(toutes ici, invisibles à cette échelle)")

ax2.set_xlim(-12, 12)
ax2.set_ylim(-12, 12)
ax2.set_aspect("equal")
ax2.set_xlabel("kpc")
ax2.set_ylabel("kpc")
ax2.set_title("Échelle : où se situe ce lot dans la Voie lactée\n"
              "(vue schématique, pas un vrai modèle galactique)", color="black")
ax2.legend(loc="upper right", fontsize=8, facecolor="white", framealpha=0.9)
ax2.tick_params(colors="black")
for spine in ax2.spines.values():
    spine.set_color("black")

plt.tight_layout()
plt.savefig("carte_galactique.png", dpi=150, bbox_inches="tight")
print("Carte sauvegardée dans carte_galactique.png")
