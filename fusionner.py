"""
Fusionne deux versions de resultats_candidats.csv par identifiant TIC
(colonne "nom"), plutôt que ligne à ligne comme le ferait Git. Utilisé
par le workflow GitHub Actions pour combiner la progression de deux
runs qui se sont chevauchés, sans jamais produire de conflit Git.

Usage :
    python3 fusionner.py distant.csv local.csv fusionne.csv
"""
import sys
import pandas as pd

distant_path, local_path, sortie_path = sys.argv[1], sys.argv[2], sys.argv[3]

distant = pd.read_csv(distant_path)
local = pd.read_csv(local_path)
fusion = pd.concat([distant, local]).drop_duplicates(subset="nom", keep="last")
fusion.to_csv(sortie_path, index=False)
print(f"Fusion : {len(distant)} distant + {len(local)} local -> {len(fusion)} fusionné")
