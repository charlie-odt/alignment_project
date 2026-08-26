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

    def fit_predict(self, outliers_to_analyse = None):
        # We have to leave one sequence out of the training set, and repeat it for each sequence.
        if outliers_to_analyse is None:
            for i in range(self._n):
                self.fit_predict_one(i)
    
    def fit_predict_one(self, index_outlier, delta = 0.05, verbose = True):
        """
        Returns True if the sequence is an outlier, False otherwise.
        """
        # 1. Remove the outlier from the dataset and create the similarity matrix
        tmp_df = self._df.drop(index_outlier)
        K_train = create_similarity_matrix(tmp_df)

        # 2. Compute the mean of the distance matrix (without outlier)
        moy = np.mean(K_train)

        # Compute the values of f for the sequences in the training set
        f_train = [K_train[i, i] - 2 * (1/(self._n - 1)) * np.sum(K_train[i, :]) + moy for i in range(K_train.shape[0])]

        # 3. Compute the threshold for outlier detection
        c = 0.125 + 2.0 * np.sqrt(moy)
        rad = (2.0 * c) / np.sqrt(self._n - 1) # with LaTeX
        threshold = np.mean(f_train) + rad + np.sqrt(np.log(1 / delta) / (2 * (self._n - 1)))

        # 4. Identify outliers

        # Compute the values of f for the sequences in the dataset with outliers
        f_outlier = 1 - 2 * (1/(self._n - 1)) * (np.sum(self._K[index_outlier, :])-1) + moy

        # 5. Print the results if verbose is set to True
        if verbose:
            print(f"Sequence: {self._seq_names[index_outlier]}")
            print(f"f_outlier: {f_outlier}")
            print(f"threshold: {threshold}")
        
            if f_outlier > threshold:
                print("The sequence is considered an outlier.")
            else:
                print("The sequence is NOT considered an outlier.")
        
        # 6. Return the result
        return f_outlier > threshold
      


if __name__ == "__main__":
    # Example usage
    csv_file = "../data/csv_files/training/BB20004.csv"
    classifier = outlier_classifier(csv_file)
    classifier.fit_predict_one(1)
