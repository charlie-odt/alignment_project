import os
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

def equation_kkt(epsilon_1,epsilon_global,moyenne_ref,m,L=1.0):
    epsilon_2 = epsilon_global-epsilon_1 
    
    #Pour éviter les divisions par zéro ou les logs négatifs
    if epsilon_1<=0 or epsilon_1>=epsilon_global or epsilon_2<=0:
        return 1e6 
    return (L/np.sqrt(m))*(1/(np.sqrt(np.log(1/epsilon_2))*epsilon_2))-(moyenne_ref/(epsilon_1**2))



def optimiser_epsilons(epsilon_global,moyenne_ref,m,L=1.0):
   
    #Résolution numérique de l'équation KKT
    estimation_initiale = epsilon_global / 2
    try:
        eps_1_opt = fsolve(equation_kkt, x0=estimation_initiale, args=(epsilon_global, moyenne_ref, m, L))[0]
        eps_2_opt = epsilon_global - eps_1_opt
        
        # Vérification des contraintes mathématiques
        if 0<eps_1_opt<epsilon_global:
            return eps_1_opt,eps_2_opt
    except:
        pass
    
    #Solution de repli en cas d'échec du solveur
    return estimation_initiale, epsilon_global - estimation_initiale


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
    
    #Transformation avec exponentielle
    #ecart = 1.0 - similarité
    ecarts = 1.0 - X_train

    
    
    #moyenne de référence transformée (somme globale sans la trace)
    somme_totale_exp = np.sum(ecarts)-np.trace(ecarts)
    moyenne_reference_exp = somme_totale_exp/(m*(m-1))
    
    #Optimisation des hyperparamètres epsilon_1 et epsilon_2 via KKT
    eps_1,eps_2 = optimiser_epsilons(EPSILON_GLOBAL, moyenne_reference_exp, m, L)
    
    #Calcul de la borne supérieure théorique
    borne_hoeffding = L*np.sqrt((4*np.log(1/eps_2))/m)
    borne_markov = (1/eps_1)*moyenne_reference_exp
    borne_totale = borne_hoeffding + borne_markov
    
    # 4. Évaluation de la protéine cible (X_test)
    ecart_cible = 1.0-np.mean(X_test)
    X_exp = np.exp(-ecart_cible / T)
    
    #Règle de décision : Si X_exp dépasse la borne_totale, c'est un extreme
    score_aberration = X_exp - borne_totale
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
