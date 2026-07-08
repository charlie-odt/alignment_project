from sklearn import svm
import numpy as np
import pandas as pd

chemin_matrice = "../../data/similarity_tab/similarity_e_maj.npy"
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

    one = svm.OneClassSVM(kernel="rbf", gamma= 0.001, nu=0.05)
    one.fit(X_train)

    prediction = one.predict(X_test)[0]
    score_distance = one.decision_function(X_test)[0]
    resultat.append({"proteine": proteine, "statut": "extreme" if prediction == -1 else "bonne", "score" : round(score_distance,2)})


df_svm = pd.DataFrame(resultat).sort_values(by="score", ascending=True)

print("resultat one class svm")
print(df_svm.head(5))
