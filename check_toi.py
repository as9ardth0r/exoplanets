"""
Vérifie si un TIC précis apparaît déjà dans la table TOI (TESS Objects
of Interest) de l'archive d'exoplanètes de la NASA -- et si oui, son
statut (PC=candidat, CP=planète confirmée, FP=faux positif, etc).

Installation :
    pip install astroquery

Usage :
    python3 check_toi.py 88507544
"""

import sys
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

tic_id = sys.argv[1] if len(sys.argv) > 1 else sys.exit("Usage : python3 check_toi.py <tic_id>")

table = NasaExoplanetArchive.query_criteria(
    table="toi",
    select="toi,tid,tfopwg_disp,pl_orbper,pl_trandep,pl_rade",
    where=f"tid={tic_id}",
)

if len(table) == 0:
    print(f"TIC {tic_id} : absent de la table TOI -- pas encore repéré par le pipeline officiel de TESS.")
else:
    print(f"TIC {tic_id} : {len(table)} entrée(s) TOI trouvée(s)\n")
    disp_connus = {
        "PC": "Candidat planétaire (pas encore confirmé)",
        "APC": "Candidat ambigu",
        "CP": "Planète confirmée",
        "KP": "Planète déjà connue",
        "FP": "Faux positif",
        "FA": "Fausse alerte",
    }
    for row in table:
        disp = str(row["tfopwg_disp"])
        print(f"  TOI-{row['toi']}  statut: {disp} ({disp_connus.get(disp, 'inconnu')})")
        print(f"    période officielle: {row['pl_orbper']}  profondeur: {row['pl_trandep']}  rayon: {row['pl_rade']}")
