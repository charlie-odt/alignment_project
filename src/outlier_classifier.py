import pandas as pd
import numpy as np
import os
from similarity import create_similarity_matrix#, identity_ratio

class outlier_classifier:
    def __init__(self, csv_file : str):
        self._df = pd.read_csv(csv_file)
        self._n = self._df.shape[0]
        self._seq_names = self._df['protein_name'].values
        self._K = create_similarity_matrix(self._df)

    def fit_predict(self, delta = 0.05, verbose = False):
        #We have to leave one sequence out of the training set, and repeat it for each sequence.
        results = []
        for i in range(self._n):
            is_outlier, f_val, thresh = self.fit_predict_one(i, delta=delta, verbose=verbose)
            results.append({
                "protein_name": self._seq_names[i],
                "is_outlier": is_outlier,
                "score": round(f_val - thresh, 4),
                "f_outlier": round(f_val, 4),
                "threshold": round(thresh, 4),
            })
        return pd.DataFrame(results).sort_values(by="score", ascending=False)
    
    def fit_predict_one(self, index_outlier, delta = 0.05, verbose = True):
        """
        Returns True if the sequence is an outlier, False otherwise.
        """
        #Remove the outlier from the dataset and create the similarity matrix
        K_train = np.delete(
            np.delete(self._K, index_outlier, axis=0), index_outlier, axis=1
        )
        m = self._n - 1
        #Compute the mean of the distance matrix (without outlier)
        moy = np.mean(K_train)

        #Compute the values of f for the sequences in the training set
        f_train = [K_train[i, i] - 2 * (1/m) * np.sum(K_train[i, :]) + moy for i in range(m)]

        #Compute the threshold for outlier detection
        c = 0.125 + 2.0 * np.sqrt(moy)
        rad = (2.0 * c) / np.sqrt(m) # with LaTeX
        threshold = np.mean(f_train) + rad + np.sqrt(np.log(1 / delta) / (2 * m))

        #Identify outliers
        #Compute the values of f for the sequences in the dataset with outliers
        f_outlier = 1 - 2 * (1/(m)) * (np.sum(self._K[index_outlier, :])-1) + moy

        #Print the results if verbose is set to True
        if verbose:
            print(f"Sequence: {self._seq_names[index_outlier]}")
            print(f"f_outlier: {f_outlier:.4f}")
            print(f"threshold: {threshold:.4f}")
        
            if f_outlier > threshold:
                print("The sequence is considered an outlier.")
            else:
                print("The sequence is NOT considered an outlier.")
        
        #Return the result
        return f_outlier > threshold, f_outlier, threshold
      


if __name__ == "__main__":
    #Example usage
    csv_file = "../data/csv_files/training/BB20004.csv"
    classifier = outlier_classifier(csv_file)
    classifier.fit_predict_one(1)
