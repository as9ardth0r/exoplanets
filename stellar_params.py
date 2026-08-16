"""
Récupère le rayon, la masse et la température de l'étoile hôte depuis
le catalogue TIC -- utile pour juger si une profondeur de transit
donnée correspond à une planète plausible ou à un compagnon stellaire.

Installation :
    pip install astroquery

Usage :
    python3 stellar_params.py 150284425
"""

import sys
from astroquery.mast import Catalogs

tic_id = sys.argv[1] if len(sys.argv) > 1 else "150284425"

result = Catalogs.query_criteria(catalog="Tic", ID=tic_id)

if len(result) == 0:
    print(f"Aucune entrée trouvée pour TIC {tic_id}")
else:
    row = result[0]
    rayon = row["rad"]
    masse = row["mass"]
    teff = row["Teff"]
    print(f"TIC {tic_id}")
    print(f"Rayon stellaire  : {rayon} R_soleil")
    print(f"Masse stellaire  : {masse} M_soleil")
    print(f"Température eff. : {teff} K")

    depth = float(input("\nProfondeur de transit mesurée (ex: 0.042 pour 4.2%) : "))
    if rayon:
        rp_rsun = (depth ** 0.5) * float(rayon)
        rp_rearth = rp_rsun * 109.2  # 1 R_soleil = 109.2 R_terre
        rp_rjup = rp_rsun * 9.73     # 1 R_soleil = 9.73 R_jupiter
        print(f"\nRayon de l'objet transitant implicite : {rp_rearth:.1f} R_terre "
              f"({rp_rjup:.2f} R_jupiter)")
        if rp_rjup > 2.0:
            print("-> Trop gros pour une planète plausible : probablement une naine "
                  "rouge ou brune en transit, pas une planète.")
        elif rp_rjup > 1.3:
            print("-> À la limite haute des planètes connues (grosses géantes gazeuses "
                  "gonflées) -- pas exclu, mais à traiter avec prudence.")
        else:
            print("-> Taille compatible avec une planète.")
