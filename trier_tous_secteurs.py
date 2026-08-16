"""
Agrège les candidats "a_verifier" de TOUS les secteurs déjà scannés
(tous les resultats_S*.csv présents), exclut ceux déjà traités dans
verdicts.csv, puis vérifie automatiquement chaque nouveau candidat
dans la table TOI officielle de la NASA -- pour ne remonter à un
examen humain que ceux qui restent vraiment non classés.

Les candidats déjà répertoriés comme TOI sont ajoutés automatiquement
à verdicts.csv (avec le bon tampon selon leur statut officiel), le
même geste qu'on faisait à la main pour le secteur 108.

Installation :
    pip install pandas astroquery

Usage :
    python3 trier_tous_secteurs.py
    python3 trier_tous_secteurs.py --verdicts ../site/verdicts.csv
"""

import argparse
import glob
import os
import re
import sys
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

DISPOSITIONS = {
    "CP": ("confirm", "Planète confirmée"),
    "KP": ("confirm", "Planète connue"),
    "PC": ("candidat", "Candidat TESS officiel"),
    "APC": ("marginal", "Candidat ambigu (TESS)"),
    "FP": ("reject", "Faux positif (TESS)"),
    "FA": ("reject", "Fausse alerte (TESS)"),
}


def charger_verdicts(path):
    if not os.path.exists(path):
        return set(), []
    lignes = list(pd.read_csv(path).to_dict("records"))
    return {str(l["tic_id"]) for l in lignes}, lignes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="resultats_S*.csv")
    parser.add_argument("--verdicts", default="verdicts.csv")
    args = parser.parse_args()

    fichiers = sorted(glob.glob(args.pattern))
    if not fichiers:
        sys.exit(f"Aucun fichier ne correspond à '{args.pattern}' dans ce dossier.")

    morceaux = []
    for f in fichiers:
        m = re.search(r"S(\d+)\.csv", f)
        secteur = m.group(1) if m else "?"
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"⚠ {f} illisible ({e}), ignoré")
            continue
        df["secteur"] = secteur
        morceaux.append(df)

    tout = pd.concat(morceaux, ignore_index=True)
    total_etoiles = len(tout)

    if "verdict_auto" not in tout.columns:
        sys.exit("Colonne verdict_auto absente -- ces résultats viennent d'une "
                  "version antérieure du pipeline, sans le tri automatique.")

    a_verifier = tout[tout["verdict_auto"] == "a_verifier"].copy()
    a_verifier["tic_id"] = a_verifier["nom"].str.replace("TIC ", "", regex=False)

    # Une même étoile peut apparaître dans plusieurs secteurs qui se
    # chevauchent -- on la traite une seule fois, en gardant la trace
    # de tous les secteurs concernés plutôt que de la dupliquer.
    secteurs_par_tic = a_verifier.groupby("tic_id")["secteur"].apply(
        lambda s: ";".join(sorted(set(s), key=int)))
    a_verifier = a_verifier.drop_duplicates(subset="tic_id", keep="first")
    a_verifier["secteur"] = a_verifier["tic_id"].map(secteurs_par_tic)

    deja_traites, lignes_verdicts = charger_verdicts(args.verdicts)
    nouveaux = a_verifier[~a_verifier["tic_id"].isin(deja_traites)]

    print(f"{len(fichiers)} secteur(s) avec résultats, {total_etoiles} étoile(s) au total")
    print(f"{len(a_verifier)} candidat(s) a_verifier au total, "
          f"{len(nouveaux)} nouveau(x) (absents de {args.verdicts})\n")

    if len(nouveaux) == 0:
        print("Rien de nouveau à trier -- tout ce qui a été détecté a déjà un verdict.")
        return

    resolus_auto = []
    vraiment_nouveaux = []

    for i, row in enumerate(nouveaux.itertuples(), 1):
        tic_id = row.tic_id
        print(f"  [{i}/{len(nouveaux)}] TIC {tic_id}...", end=" ", flush=True)
        try:
            table = NasaExoplanetArchive.query_criteria(
                table="toi", select="toi,tfopwg_disp", where=f"tid={tic_id}")
        except Exception as e:
            print(f"requête échouée ({e}), laissé à vérifier")
            vraiment_nouveaux.append(row)
            continue

        if len(table) > 0:
            disp = str(table[0]["tfopwg_disp"])
            toi = table[0]["toi"]
            stamp_class, stamp_label = DISPOSITIONS.get(disp, ("marginal", disp))
            print(f"déjà classé TOI-{toi} ({stamp_label})")
            resolus_auto.append({
                "tic_id": tic_id, "stamp_class": stamp_class,
                "stamp_label": f"{stamp_label} (TOI-{toi})",
                "periode": row.periodes, "snr": row.snr_max,
                "note": f"Secteur {row.secteur}. Statut officiel TOI : {disp}. "
                        f"Classé automatiquement, à vérifier visuellement avant publication sur le site.",
            })
        else:
            print("non classé, à vérifier à l'œil")
            vraiment_nouveaux.append(row)

    if resolus_auto:
        lignes_verdicts.extend(resolus_auto)
        pd.DataFrame(lignes_verdicts).to_csv(args.verdicts, index=False)
        print(f"\n{len(resolus_auto)} candidat(s) ajouté(s) automatiquement à {args.verdicts}")

    print(f"\n{len(vraiment_nouveaux)} candidat(s) vraiment non classé(s), à trier à l'œil :")
    if vraiment_nouveaux:
        df_finale = pd.DataFrame([{
            "nom": r.nom, "secteur": r.secteur, "Tmag": getattr(r, "Tmag", "?"),
            "periodes": r.periodes, "snr_max": r.snr_max,
        } for r in vraiment_nouveaux]).sort_values("snr_max", ascending=False)
        print(df_finale.to_string(index=False))
        print("\nPour la vue en grille :")
        ids = " ".join(r.tic_id for r in vraiment_nouveaux)
        print(f"  python3 triage_rapide.py {ids}")


if __name__ == "__main__":
    main()
