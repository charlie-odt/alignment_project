from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

chemin_matrice = "../../data/similarity_tab/training/similarity_BB20003.npy"   #a changer selon le fichier quon veut regarder
matrice = np.load(chemin_matrice)
N = matrice.shape[0]
matrice_distance = 1-matrice
numeros_proteines = [f"Prot_{i}" for i in range(N)]

resultat = []
for i,proteine in enumerate(numeros_proteines):
    
    X_train = np.delete(matrice_distance,i, axis = 0)
    X_train = np.delete(X_train,i, axis = 1)

    X_test = np.delete(matrice_distance[i],i)
    X_test = X_test.reshape(1,-1)

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_train)

    prediction = model.predict(X_test)[0]
    score_distance = model.score_samples(X_test)[0]
    resultat.append({"proteine": proteine, "statut": "extreme" if prediction == -1 else "bonne", "score" : round(score_distance,2)})


df_forest = pd.DataFrame(resultat).sort_values(by="score", ascending=True)

print("resultat forest ")
print(df_forest)