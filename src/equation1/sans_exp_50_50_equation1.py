import os
import numpy as np
import pandas as pd


#essai theoreme 1

#Chargement de la matrice
chemin_matrice = "../../data/similarity_tab/test/similarity_e_maj.npy"
matrice_id = np.load(chemin_matrice)
N = matrice_id.shape[0]

#Noms des protéines
numeros_proteines = [f"Prot_{i}" for i in range(N)]

#parametres
EPSILON_GLOBAL = 0.10   #Risque erreur
L = 1.0                 #la fonction exponentielle négative dans [0, 1]
T = 0.05                #T pour calibrer la sensibilité

resultats = []

#Boucle principale : Leave-One-Out
for i, proteine in enumerate(numeros_proteines):
    
    #Extraction de la sous-matrice de référence (X_train) et de la cible (X_test)
    X_train = np.delete(matrice_id,i,axis=0)
    X_train = np.delete(X_train,i,axis=1)
    
    X_test = np.delete(matrice_id[i],i)
            
    m = X_train.shape[0]
    
    #ecart = 1.0 - similarité
    ecarts = 1.0 - X_train

    
    
    #moyenne de référence transformée (somme globale sans la trace)
    somme_totale = np.sum(ecarts)-np.trace(ecarts)
    moyenne_reference = somme_totale/(m*(m-1))
    
    #pas d'optimisation on fait 50/50
    eps_1 = EPSILON_GLOBAL/2
    eps_2 = EPSILON_GLOBAL/2
    
    #Calcul de la borne supérieure théorique
    borne_hoeffding = L*np.sqrt((4*np.log(1/eps_2))/m)
    borne_markov = (1/eps_1)*moyenne_reference
    borne_totale = borne_hoeffding + borne_markov
    
    # 4. Évaluation de la protéine cible (X_test)
    ecart_cible = 1.0-np.mean(X_test)

    
    #Règle de décision : Si X_exp dépasse la borne_totale, c'est un extreme
    score_aberration = ecart_cible- borne_totale
    statut = "extreme" if score_aberration > 0 else "bonne"
    
    #Stockage des résultats au même format
    resultats.append({
        "proteine": proteine, 
        "statut": statut, 
        "score": round(score_aberration, 2)
    })


df_final = pd.DataFrame(resultats).sort_values(by="score", ascending=False)

print(f"resultats (Température = {T} et Risque global epsilon = {EPSILON_GLOBAL})")
print(df_final.head(5).to_string(index=False))
print("borne_totale =", borne_totale)
print("borne_hoeffding =", borne_hoeffding)
print("borne_markov =",borne_markov)
print("eps_1 =",eps_1," , eps_2 =",eps_2)
