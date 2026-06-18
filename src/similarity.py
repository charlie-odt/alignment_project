import os
import pandas as pd
import numpy as np

def identity_ratio(seq1, seq2):
    """
    Computes the identity ratio between two protein sequences.
    """
    counter = 0
    length = 0
    while seq1 and seq2:
        length += 1
        if seq1[0] == seq2[0] and seq1[0] != '-':
            counter += 1
        seq1 = seq1[1:]
        seq2 = seq2[1:]
    return counter/length

def create_similarity_matrix(df):
    """
    Creates the similarity matrix from the alignment dataframe and returns the NumPy matrix.
    """
    N = df.shape[0]
    mat = np.zeros((N, N))

    for i in range(N):
        for j in range(i,N):
            mat[i,j] = identity_ratio(df.loc[i]["sequence"], df.loc[j]["sequence"])
            mat[j,i] = identity_ratio(df.loc[i]["sequence"], df.loc[j]["sequence"])
    
    return mat

if __name__ == "__main__":
    
    for name in os.listdir("../data/csv_files"):
        alt_name = name.replace(".csv", ".npy")
        alt_name = "../data/similarity_tab/similarity_" + alt_name
        
        df = pd.read_csv(f"../data/csv_files/{name}")
        mat = create_similarity_matrix(df)
        np.save(alt_name, mat)
        