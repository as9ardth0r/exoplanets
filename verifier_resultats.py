import pandas as pd
df = pd.read_csv('resultats_candidats.csv')
hits = df[df['n_candidats'] > 0].sort_values('snr_max', ascending=False)
print(f'{len(hits)} candidat(s)')
print(hits[['nom','RA','Dec','Tmag','n_candidats','periodes','snr_max']].to_string(index=False))

