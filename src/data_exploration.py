import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import TRAIN_CSV

def explore_data(csv_path='data/train.csv'):
    """
    Charge le fichier CSV, affiche les statistiques descriptives 
    et génère un graphique de la distribution des classes.
    """

    if not os.path.exists(csv_path):
        print(f"Erreur : Le fichier '{csv_path}' est introuvable dans le dossier actuel.")
        return

    print(f"Chargement des données depuis '{csv_path}'...")
    df = pd.read_csv(csv_path)

    # Informations générales
    print("\n--- APERÇU DES DONNÉES ---")
    print(df.head())
    print(f"\nNombre total d'images d'entraînement : {len(df)}")
    print(f"Nombre de classes uniques : {df['Class'].nunique()}")

    # Calcul de la fréquence de chaque classe
    class_counts = df['Class'].value_counts()

    # Identification des classes surreprésentées et sous-représentées
    print("\n--- TOP 10 DES CLASSES SURREPRÉSENTÉES ---")
    print(class_counts.head(10))

    print("\n--- TOP 10 DES CLASSES SOUS-REPRÉSENTÉES ---")
    print(class_counts.tail(10))

    # Visualisation du déséquilibre
    print("\nGénération du graphique de distribution...")
    plt.figure(figsize=(12, 6))

    # Tracer un diagramme en barres
    sns.barplot(
        x=range(len(class_counts)), 
        y=class_counts.values, 
        color="royalblue"
    )

    # Ajout d'une ligne pour la moyenne
    mean_images = class_counts.mean()
    plt.axhline(
        mean_images, 
        color='red', 
        linestyle='dashed', 
        linewidth=2, 
        label=f'Moyenne ({mean_images:.1f} images/classe)'
    )

    plt.title("Distribution du nombre d'images par classe (Les 315 classes)", fontsize=14)
    plt.xlabel("Classes (triées de la plus à la moins fréquente)", fontsize=12)
    plt.ylabel("Nombre d'images", fontsize=12)

    # Masquer les noms des classes sur l'axe X (illisible avec 315 noms)
    plt.xticks([]) 
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    # 5. Sauvegarde de l'image pour le compte-rendu TP
    output_image = "output/distribution_classes.png"
    plt.savefig(output_image, dpi=300)
    print(f"Graphique sauvegardé avec succès sous le nom : '{output_image}'")
    
    # 6. Affichage interactif à l'écran
    plt.show()

