import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

def mds(dossier_sources,dossier_sortie="data/MDS"):

    #Création du dossier de destination s'il n'existe pas encore
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)
        
    #Liste les fichiers du dossier
    fichiers = [f for f in os.listdir(dossier_sources) if f.endswith(".npy")]

    if not fichiers:
        print(f"Aucune matrice trouvée dans : {dossier_sources}")
        return

    for nom_fichier in fichiers:
        chemin_matrice = os.path.join(dossier_sources,nom_fichier)
        matrice_similarite = np.load(chemin_matrice)
        
        #Vérification du format de la matrice
        if matrice_similarite.ndim != 2 or matrice_similarite.shape[0] != matrice_similarite.shape[1]:
            print(f"le fichier {nom_fichier} est une matrice non carrée")
            continue
            
        #passage de la similarite a la distance 
        matrice_distance = 1.0-matrice_similarite
        
        #MDS
        mds = MDS(n_components=2,dissimilarity="precomputed",random_state=42,n_init=4)
        coordonnees_2d = mds.fit_transform(matrice_distance)
        
        # Génération de la figure
        plt.figure(figsize=(10,8))
        
        plt.scatter(
            coordonnees_2d[:,0], 
            coordonnees_2d[:,1], 
            color="blue", 
            alpha=0.7, 
            edgecolors="black", 
            s=60
        )
        
        #Nettoyage et mise en forme du titre
        nom_propre = nom_fichier.replace(".npy","").replace("similarity_","")
        plt.title(f"Projection MDS : {nom_propre}",fontsize=13,fontweight='bold',pad=12)
        plt.xlabel("Dimension MDS 1",fontsize=11)
        plt.ylabel("Dimension MDS 2",fontsize=11)
        plt.grid(True,linestyle="--",alpha=0.5)
        
        plt.axhline(0,color="grey",linewidth=0.5,alpha=0.5)
        plt.axvline(0,color="grey",linewidth=0.5,alpha=0.5)
        
        # Exportation haute résolution pour LaTeX
        nom_image = f"mds_{nom_propre}.png"
        chemin_sauvegarde = os.path.join(dossier_sortie, nom_image).replace("\\", "/")
        plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"MDS terminé : {chemin_sauvegarde}")

    print("\nTous les graphiques ont été générés ")

#a mettre dans le main si on veut tout regrouper 
if __name__ == "__main__":
    dossier_input = "data/similarity_tab"
    mds(dossier_input)