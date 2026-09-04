import os
import numpy as np
import pandas as pd
from outlier_classifier import outlier_classifier

#Chemins d'accès
dossier_entree = "../data/csv_files/test"
dossier_sortie = "../data/results_test"
os.makedirs(dossier_sortie, exist_ok=True)

#Récupération des fichiers .csv
fichiers = [f for f in os.listdir(dossier_entree) if f.endswith(".csv")]
print(f"Nombre d'alignements trouvés : {len(fichiers)}")

resultats_globaux = []

#Boucle sur chaque alignement
for idx, nom_fichier in enumerate(fichiers, start=1):
    chemin_csv = os.path.join(dossier_entree, nom_fichier)

    #Instanciation de la classe
    classifier = outlier_classifier(chemin_csv)

    #Exécution du Leave-One-Out sur tout le fichier
    df_fichier = classifier.fit_predict(delta=0.05, verbose=False) #on modifie ici le delta 
    df_fichier["fichier_alignement"] = nom_fichier

    #Sauvegarde individuelle
    nom_csv_sortie = os.path.join(dossier_sortie, f"res_{nom_fichier}")
    df_fichier.to_csv(nom_csv_sortie, index=False)

    resultats_globaux.append(df_fichier)

#Récapitulatif global
if resultats_globaux:
    df_total = pd.concat(resultats_globaux, ignore_index=True)
    chemin_global = os.path.join(dossier_sortie, "recapitulatif_global.csv")
    df_total.to_csv(chemin_global, index=False)
    
    #Résumé
    total_sequences = len(df_total)
    total_outliers = df_total["is_outlier"].sum()
    print(f"Séquences analysées : {total_sequences}")
    print(f"Anomalies détectées : {total_outliers}")
