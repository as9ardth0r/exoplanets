"""
Compte le nombre de signaux de transit candidats par étoile, sans
génération de graphique — juste un fichier de sortie récapitulatif
avec nom, coordonnées et nombre de candidats.

Principe : pour chaque étoile, recherche BLS -> si le signal est
significatif, on masque ce transit et on recherche à nouveau -> on
répète jusqu'à ce qu'il n'y ait plus de signal significatif (ou
jusqu'à --max-planets). Un pic collé au bord de la plage de périodes
testée est ignoré (presque toujours un faux positif).

Traite plusieurs étoiles en parallèle (--workers) et reprend
automatiquement où il s'était arrêté si tu relances la même commande
avec le même --output : les étoiles déjà présentes dans le fichier de
sortie sont sautées.

ATTENTION : le nombre obtenu n'est PAS une confirmation d'exoplanète.
C'est un compte de signaux périodiques candidats détectés par
l'algorithme — des étoiles variables ou des binaires à éclipses
peuvent produire de faux positifs. Une vraie confirmation demande un
examen visuel de la courbe repliée, voire un suivi indépendant.

Installation :
    pip install lightkurve pandas

Usage :
    python3 exoplanet_bls_count.py all_targets_S108_v1.csv --limit 20
    python3 exoplanet_bls_count.py all_targets_S108_v1.csv --limit 13000 --workers 4

Pour un run long, lance en arrière-plan avec nohup ou dans un tmux :
    nohup python3 exoplanet_bls_count.py all_targets_S108_v1.csv --limit 13000 --workers 4 > log.txt 2>&1 &
Relancer EXACTEMENT la même commande reprend automatiquement là où ça s'était arrêté.
"""

import argparse
import os
import time
import logging
import tempfile
import warnings
import threading
import concurrent.futures
import numpy as np
import pandas as pd
import lightkurve as lk
from astropy.units import UnitsWarning

# Ce message ("period contains N points... consider frequency_factor")
# vient du module logging de lightkurve (log.warning), pas du module
# warnings standard -- il faut donc relever le niveau du logger
# "lightkurve" pour le faire taire. Confirmé sans risque : c'est une
# estimation cosmétique qui n'affecte pas le calcul réel, puisqu'on
# fournit notre propre grille de périodes (celle-ci est alors utilisée
# directement, sans jamais passer par l'estimation qui déclenche ce
# message).
logging.getLogger("lightkurve").setLevel(logging.ERROR)

# Idem pour les avertissements astropy sur les unités FITS non-standard
# ('e-/s', 'pixels', 'BJD - 2457000, days') : cosmétique, sans impact
# sur les valeurs numériques utilisées par le pipeline.
warnings.filterwarnings("ignore", category=UnitsWarning)

COLONNES = ["nom", "RA", "Dec", "Tmag", "n_candidats", "periodes", "snr_max", "verdict_auto", "raisons_auto", "n_secteurs", "secondes", "status"]


def evaluer_signal(lc, bls, best_period, best_t0, best_duration, best_depth, tic_id):
    """Contrôles au-delà des 5 filtres de base, qui approchent numériquement
    ce qu'on repérait à l'œil sur les graphiques : variabilité continue,
    éclipse secondaire, compagnon trop gros pour une planète, signal
    encore montant au bord de la plage testée.

    Renvoie (verdict, [raisons]). verdict vaut "rejete" ou "a_verifier" --
    JAMAIS "confirme" : une confirmation ne s'automatise pas, elle demande
    un suivi indépendant hors du champ de ce pipeline.
    """
    raisons = []

    # 1. Variabilité continue : le bruit hors-transit est-il déjà du
    #    même ordre que la profondeur du signal repéré ?
    try:
        folded = lc.fold(period=best_period, epoch_time=best_t0, normalize_phase=True)
        phase = folded.phase.value
        flux = folded.flux.value
        demi_duree_frac = 1.5 * (best_duration.value / best_period.value)
        hors_transit = np.abs(phase) > demi_duree_frac
        depth = float(best_depth)
        if hors_transit.sum() > 20 and depth > 0:
            bruit_hors_transit = np.nanstd(flux[hors_transit])
            if bruit_hors_transit / depth > 0.4:
                raisons.append("variabilite_continue")
    except Exception:
        pass

    # 2. Éclipse secondaire : un second creux ailleurs en phase, une
    #    fois le principal masqué ?
    try:
        mask_primaire = bls.get_transit_mask(period=best_period, transit_time=best_t0,
                                              duration=best_duration)
        folded2 = lc[~mask_primaire].fold(period=best_period, epoch_time=best_t0,
                                           normalize_phase=True)
        bins = np.linspace(-0.5, 0.5, 41)
        idx = np.digitize(folded2.phase.value, bins)
        flux2 = folded2.flux.value
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            moyennes = [np.nanmean(flux2[idx == i]) for i in range(1, len(bins))]
        moyennes = [m for m in moyennes if not np.isnan(m)]
        if moyennes and depth > 0:
            profondeur_secondaire = 1.0 - min(moyennes)
            if profondeur_secondaire > 0.25 * depth:
                raisons.append("eclipse_secondaire")
    except Exception:
        pass

    # 3. Rayon implicite de l'objet en transit, via le rayon stellaire
    #    du catalogue TIC -- nécessite le réseau, échoue silencieusement
    #    sinon (on ne rejette jamais faute d'avoir pu vérifier).
    try:
        from astroquery.mast import Catalogs
        res = Catalogs.query_criteria(catalog="Tic", ID=str(tic_id))
        rayon_etoile = float(res[0]["rad"])
        rp_rjup = (depth ** 0.5) * rayon_etoile * 9.73
        if rp_rjup > 2.0:
            raisons.append(f"compagnon_trop_gros_{rp_rjup:.1f}Rjup")
    except Exception:
        pass

    # 4. Puissance encore montante près du bord de la plage testée : la
    #    vraie période est probablement hors de la plage explorée.
    #    (Contrôle testé et retiré : une simple comparaison de tendance
    #    déclenchait aussi sur des signaux propres, loin du bord --
    #    remplacé par une marge de bord élargie directement dans
    #    compter_candidats, qui s'est avérée plus fiable.)

    verdict = "rejete" if raisons else "a_verifier"
    return verdict, raisons


def compter_candidats(tic_id, mission, period_min, period_max,
                       snr_threshold=10, max_planets=5, n_periods=5000,
                       max_sectors=10):
    # exptime fixe la cadence : sans ça, TESS peut renvoyer un mélange
    # de produits 2 minutes ET 20 secondes ("fast cadence"), ce qui
    # multiplie le nombre de points et ralentit le calcul.
    if mission == "Kepler":
        author, exptime = "Kepler", "long"
    else:
        author, exptime = "SPOC", 120
    search_result = lk.search_lightcurve(f"TIC {tic_id}", author=author, exptime=exptime)

    if len(search_result) == 0:
        return {"status": "aucune_donnee", "n_candidats": 0, "periodes": "", "snr_max": "",
                "verdict_auto": "", "raisons_auto": "", "n_secteurs": 0}

    n_secteurs = len(search_result)
    if n_secteurs > max_sectors:
        # Une étoile avec énormément de secteurs disponibles peut faire
        # traîner le téléchargement très longtemps ; on se limite aux
        # N premiers secteurs, largement suffisant pour un transit de
        # période courte.
        search_result = search_result[:max_sectors]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Téléchargement dans un dossier temporaire propre à cette
        # étoile : les fichiers sont automatiquement supprimés en
        # sortant de ce bloc, ce qui évite d'accumuler des dizaines de
        # Go dans le cache permanent de lightkurve (~/.lightkurve/cache)
        # sur un run de plusieurs milliers d'étoiles.
        lc_collection = search_result.download_all(download_dir=tmpdir)
        lc_courante = lc_collection.stitch().flatten(window_length=901).remove_outliers()

        periodes_trouvees = []
        snr_trouvees = []
        verdict_auto, raisons_auto = "", ""

        for _ in range(max_planets):
            if len(lc_courante) < 50:
                break  # plus assez de points après les masquages successifs

            period_grid = np.linspace(period_min, period_max, n_periods)
            bls = lc_courante.to_periodogram(method="bls", period=period_grid, frequency_factor=2000)
            snr_proxy = float(bls.max_power / np.median(bls.power))

            if snr_proxy < snr_threshold:
                break  # plus de signal significatif au-dessus du bruit

            best_period = bls.period_at_max_power
            best_t0 = bls.transit_time_at_max_power
            best_duration = bls.duration_at_max_power
            best_depth = bls.depth_at_max_power

            # Une vraie planète creuse rarement plus de ~5% de flux,
            # même dans les cas extrêmes (grosse géante gazeuse autour
            # d'une petite étoile). Au-delà, c'est presque toujours un
            # compagnon de taille stellaire -- binaire à éclipses ou
            # variation ellipsoïdale -- pas une planète.
            if float(best_depth) > 0.05:
                break

            # Un pic collé au bord de la plage de périodes testée n'est
            # presque jamais un vrai transit : c'est le signe d'un signal
            # dont la vraie période tombe hors de cette plage. On l'ignore
            # et on arrête, plutôt que de le compter comme un candidat.
            # Marge à 3% (plutôt que 2%) : TIC 54002166 (14.6955j sur une
            # plage 0.5-15j) ratait de 21 minutes une marge à 2%, alors
            # que c'était bien un artefact de bord.
            marge = 0.03 * (period_max - period_min)
            au_bord = (best_period.value - period_min < marge) or (period_max - best_period.value < marge)
            if au_bord:
                break

            # Une vraie planète transite rarement plus de ~10-15% de sa
            # période. Un ratio durée/période beaucoup plus grand indique
            # presque toujours de la variabilité stellaire continue
            # (pulsation, effet ellipsoïdal d'une binaire serrée) que BLS
            # a mal interprétée comme un transit en "boîte".
            ratio_duree = best_duration.value / best_period.value
            if ratio_duree > 0.15:
                break

            # Si BLS retrouve presque la même période après masquage,
            # c'est que le masquage n'a pas retiré le signal -- typique
            # d'une variabilité stellaire continue (pulsation, taches)
            # qu'un masquage de transit ponctuel n'efface pas. Ça remet
            # aussi en cause la toute première détection : on invalide
            # l'étoile entière plutôt que de juste arrêter de compter.
            if any(abs(best_period.value - p) < 0.01 for p in periodes_trouvees):
                periodes_trouvees = []
                break

            # ~13.7j est la période orbitale de TESS elle-même : chaque
            # secteur contient une coupure de données à mi-parcours
            # (téléchargement au périgée). Empilée sur plusieurs
            # secteurs, cette coupure récurrente peut mimer un faux
            # signal périodique à cette période précise.
            if abs(best_period.value - 13.7) < 0.9:
                break

            periodes_trouvees.append(round(float(best_period.value), 4))
            snr_trouvees.append(round(snr_proxy, 1))

            if not verdict_auto:
                # Seulement sur le premier signal retenu : les contrôles
                # 1-4 coûtent cher (dont une requête réseau), pas la
                # peine de les répéter sur les signaux suivants du
                # même astre.
                verdict_auto, raisons = evaluer_signal(
                    lc_courante, bls, best_period, best_t0, best_duration, best_depth, tic_id)
                raisons_auto = ";".join(raisons)

            mask = bls.get_transit_mask(period=best_period, transit_time=best_t0,
                                         duration=best_duration)
            lc_courante = lc_courante[~mask]

    return {
        "status": "ok",
        "n_candidats": len(periodes_trouvees),
        "periodes": ";".join(str(p) for p in periodes_trouvees),
        "snr_max": max(snr_trouvees) if snr_trouvees else "",
        "n_secteurs": n_secteurs,
        "verdict_auto": verdict_auto,
        "raisons_auto": raisons_auto,
    }


def traiter_une_cible(tic_id, ra, dec, tmag, args):
    debut = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                compter_candidats, tic_id, args.mission, args.period_min, args.period_max,
                snr_threshold=args.snr_threshold, max_planets=args.max_planets,
                max_sectors=args.max_sectors,
            )
            res = future.result(timeout=args.timeout)
    except concurrent.futures.TimeoutError:
        res = {"status": "timeout", "n_candidats": 0, "periodes": "", "snr_max": "",
               "verdict_auto": "", "raisons_auto": "", "n_secteurs": "?"}
    except Exception as e:
        res = {"status": "erreur", "n_candidats": 0, "periodes": "", "snr_max": "",
               "verdict_auto": "", "raisons_auto": "", "n_secteurs": "?", "erreur": str(e)}
    duree = time.perf_counter() - debut
    res["nom"] = f"TIC {tic_id}"
    res["RA"] = ra
    res["Dec"] = dec
    res["Tmag"] = tmag
    res["secondes"] = round(duree, 1)
    return res


def main():
    parser = argparse.ArgumentParser(
        description="Compte les signaux de transit candidats par étoile (sans graphique)")
    parser.add_argument("csv_file", help="CSV avec colonnes TICID, RA, Dec (ex: liste officielle TESS)")
    parser.add_argument("--mission", default="TESS", choices=["Kepler", "TESS"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--mag-max", type=float, default=None,
                         help="Filtrer sur Tmag < cette valeur, si la colonne existe")
    parser.add_argument("--mag-min", type=float, default=4.0,
                         help="Exclut les étoiles plus brillantes que ça (saturation du capteur)")
    parser.add_argument("--max-sectors", type=int, default=10,
                         help="Nombre max de secteurs téléchargés par étoile")
    parser.add_argument("--period-min", type=float, default=0.5)
    parser.add_argument("--period-max", type=float, default=15)
    parser.add_argument("--snr-threshold", type=float, default=10,
                         help="Seuil puissance/bruit en dessous duquel on arrête de chercher")
    parser.add_argument("--max-planets", type=int, default=5,
                         help="Nombre maximal de signaux recherchés par étoile")
    parser.add_argument("--workers", type=int, default=1,
                         help="Étoiles traitées en parallèle. Défaut prudent à 1 : un run précédent "
                              "s'est bloqué avec --workers 4, probablement un verrou de cache astropy "
                              "partagé entre threads. Remonte progressivement une fois confirmé stable.")
    parser.add_argument("--timeout", type=float, default=120,
                         help="Secondes max par étoile avant d'abandonner et de passer à la suivante")
    parser.add_argument("--output", default="resultats_candidats.csv")
    args = parser.parse_args()

    targets = pd.read_csv(args.csv_file, comment="#")
    if args.mag_max is not None and "Tmag" in targets.columns:
        targets = targets[targets["Tmag"] < args.mag_max]
    if args.mag_min is not None and "Tmag" in targets.columns:
        avant = len(targets)
        targets = targets[targets["Tmag"] >= args.mag_min]
        exclues = avant - len(targets)
        if exclues:
            print(f"{exclues} étoile(s) trop brillante(s) exclue(s) (Tmag < {args.mag_min})")
    if "Tmag" in targets.columns:
        targets = targets.sort_values("Tmag")
    targets = targets.head(args.limit)

    # Reprise : si le fichier de sortie existe déjà, on charge ce qui a
    # déjà été traité et on saute ces étoiles-là.
    resultats = []
    deja_traites = set()
    if os.path.exists(args.output):
        try:
            deja_df = pd.read_csv(args.output)
            resultats = deja_df.to_dict("records")
            deja_traites = set(deja_df["nom"].astype(str))
            print(f"Reprise : {len(deja_traites)} étoile(s) déjà traitée(s) dans {args.output}, sautées")
        except Exception as e:
            print(f"Impossible de lire {args.output} existant ({e}) — on repart de zéro")

    targets = targets[~targets["TICID"].apply(lambda t: f"TIC {t}" in deja_traites)]

    print(f"{len(targets)} cible(s) restant(s) à traiter, {args.workers} en parallèle\n")

    verrou = threading.Lock()

    def sauvegarder():
        pd.DataFrame(resultats)[COLONNES].to_csv(args.output, index=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for row in targets.itertuples():
            fut = executor.submit(traiter_une_cible, row.TICID,
                                   getattr(row, "RA", None), getattr(row, "Dec", None),
                                   getattr(row, "Tmag", None), args)
            futures[fut] = row.TICID

        termines = 0
        for future in concurrent.futures.as_completed(futures):
            tic_id = futures[future]
            try:
                res = future.result()
            except Exception as e:
                res = {"nom": f"TIC {tic_id}", "status": "erreur", "n_candidats": 0,
                       "periodes": "", "snr_max": "", "verdict_auto": "", "raisons_auto": "",
                       "n_secteurs": "?", "secondes": 0,
                       "RA": None, "Dec": None, "Tmag": None, "erreur": str(e)}

            with verrou:
                termines += 1
                resultats.append(res)
                sauvegarder()
                print(f"[{termines}/{len(targets)}] TIC {tic_id}: {res.get('status')} — "
                      f"{res.get('n_candidats', 0)} candidat(s) ({res.get('secondes', '?')}s)")

    df = pd.DataFrame(resultats)
    print(f"\nRésultats sauvegardés dans {args.output} ({len(df)} étoile(s) au total)")

    hits = df[df["n_candidats"] > 0].sort_values("n_candidats", ascending=False)
    print(f"\nÉtoiles avec au moins un candidat ({len(hits)}/{len(df)}) :")
    if len(hits):
        print(hits[["nom", "RA", "Dec", "n_candidats", "periodes"]].to_string(index=False))
    else:
        print("Aucune sur ce lot.")


if __name__ == "__main__":
    main()
