import numpy as np 
import pandas as pd 

chemin_matrice = "../data/similarity_tab/test/similarity_e_maj.npy"
matrice = np.load(chemin_matrice)

N = matrice.shape[0]

numeros_proteines = [f"Prot_{i}" for i in range(N)]

DELTA = 0.0005  #risque
resultats = []

for i, proteine in enumerate(numeros_proteines):

    X_train = np.delete(matrice,i, axis = 0)
    X_train = np.delete(X_train,i, axis = 1)

    X_test = np.delete(matrice[i],i)
    
    m = X_train.shape[0]

    #premier terme
    #distances au carré dans le RKHS pour le groupe train
    norme_O_sq = np.mean(X_train)   # on calcule le ||O||**2 pour calculer f(Xi)
    #distance de chaque élément du train par rapport au centre O
    f_train = norme_O_sq + 1.0 - 2.0 * np.mean(X_train, axis=1)    #f(Xi)
    moyenne_empirique_f = np.mean(f_train)    #la moyenne pour obtenir le premier terme 
    
    
    #distance de la protéine cible (avec le centre O) qu'on va comparer a la borne
    f_cible = norme_O_sq + 1.0 - 2.0 * np.mean(X_test)


    #deuxieme terme
    #calcul de la constante c 
    c = 0.125 + 2.0 * np.sqrt(norme_O_sq)
    

    terme_rademacher = (2.0 * c) / np.sqrt(m) #avec le lateX

    #troisieme terme
    terme_confiance = np.sqrt(np.log(1.0 / DELTA) / (2.0 * m))
    
    borne_totale = moyenne_empirique_f + terme_rademacher + terme_confiance
    
    #règle de décision   
    score = f_cible - borne_totale
    statut = "extreme" if score > 0 else "bonne"

    resultats.append({
        "proteine": proteine,
        "statut": statut,
        "score": round(score, 4),
        "f_cible": round(f_cible, 4),
        "borne": round(borne_totale, 4)
    })

df = pd.DataFrame(resultats).sort_values(by="score", ascending=False)
print(df.head(10))





