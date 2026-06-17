
import os
import numpy as np
import pandas as pd
#pour recuperer alignement 
def lire_fichier(nom_fichier):
    try:
        with open(nom_fichier, "r",encoding = "utf-8") as fichier:
            nom_proteine = None
            sequence = []
            contenu = {}

            for ligne in fichier:
                ligne = ligne.strip()
                if ligne.startswith(">"):
                    if nom_proteine:
                        contenu[nom_proteine] = "".join(sequence)
                        sequence = []
                    nom_proteine = ligne[1:]
                else:
                    sequence.append(ligne)
            if nom_proteine:
                contenu[nom_proteine] = "".join(sequence)
        return contenu

    except FileNotFoundError:
        print(f"Le fichier '{nom_fichier}' n'a pas été trouvé.")
        return None
    

#pour calcul identite 

def calcul_identite(seq1,seq2):
  positions_identiques= 0
  longueur_utile = 0
    for aa1,aa2 in zip(seq1,seq2):
        if aa1 == '-' and aa2 == '-':
          continue
        longueur_utile += 1
        if aa1 == aa2:
          positions_identiques += 1
    if longueur_utile == 0:
      return 0.0
  return (positions_identiques / longueur_utile) * 100

#lance le calcul et creation des fichiers avec les matrices d'identite
dossier = "alignements_fasta"
dossier_destination = "matrices_resultats"

if not os.path.exists(dossier_destination):
    os.makedirs(dossier_destination)

fichiers = [f for f in os.listdir(dossier) if f.lower().endswith(".fasta")]

for nom_fichier in fichiers:
  chemin = os.path.join(dossier, nom_fichier)
  sequences= lire_fichier(chemin)
  if not sequences:
    continue
  noms_proteines = list(sequences.keys())
  N = len(noms_proteines)

  matrice_id = np.zeros((N, N))
  for i in range(N):
    for j in range(i, N):
      score =calcul_identite(sequences[noms_proteines[i]],sequences[noms_proteines[j]])
      matrice_id[i, j] = score
      matrice_id[j, i] = score

  df_matrice = pd.DataFrame(matrice_id, index=noms_proteines, columns=noms_proteines)
  noms_fichiers = nom_fichier.lower().replace(".fasta", "_identite.csv")

  chemin_sauvegarde = os.path.join(dossier_destination, noms_fichiers)
  
  df_matrice.to_csv(chemin_sauvegarde)
