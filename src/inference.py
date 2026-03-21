import os
import glob
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.model import get_alexnet_model
from config import TRAIN_CSV, TEST_CSV, TEST_IMG_DIR, MODEL_WEIGHTS, SUBMISSION_FILE

def load_idx_to_class(csv_path=TRAIN_CSV):
    """Reconstruit le dictionnaire de mapping entier -> texte."""
    df = pd.read_csv(csv_path)
    unique_classes = sorted(df['Class'].unique())
    idx_to_class = {idx: cls_name for idx, cls_name in enumerate(unique_classes)}
    return idx_to_class

def generate_submission(model_path='best_alexnet_model.pth', 
                        test_csv='test.csv', 
                        test_dir='test', 
                        output_file='submission.csv'):
    
    # Configuration de l'appareil
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inférence lancée sur : {device}")

    # Chargement du modèle et des poids
    model = get_alexnet_model(num_classes=315, pretrained=False) #False : poids déjà entraînés
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Modèle chargé depuis '{model_path}'.")
    except FileNotFoundError:
        print(f"Erreur : Le fichier de poids '{model_path}' est introuvable.")
        print("Avez-vous lancé l'entraînement avec 'train.py' ?")
        return
        
    model = model.to(device)
    model.eval()

    # Récupération du dictionnaire de classes
    idx_to_class = load_idx_to_class(TRAIN_CSV)
    
    try:
        test_df = pd.read_csv(test_csv)
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{test_csv}' est introuvable.")
        return

    # Définition des transformations (identiques à la validation)
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # Boucle d'inférence sur les images de test
    print(f"Génération des prédictions pour {len(test_df)} images...")
    predictions = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_df))):
            img_id = str(test_df.loc[idx, 'Id'])
            
            search_pattern = os.path.join(test_dir, f"{img_id}.*")
            matching_files = glob.glob(search_pattern)
            
            if not matching_files:
                print(f"\nAvertissement : Aucune image trouvée pour l'ID de test {img_id}")
                predicted_class_name = idx_to_class[0]
            else:
                img_path = matching_files[0]
                try:
                    image = Image.open(img_path).convert("RGB")
                    input_tensor = transform(image)
                    input_batch = input_tensor.unsqueeze(0).to(device)
                    
                    output = model(input_batch)
                    _, predicted_idx = torch.max(output, 1)
                    predicted_class_name = idx_to_class[predicted_idx.item()]
                    
                except Exception as e:
                    print(f"\nErreur sur l'image {img_id}: {e}")
                    predicted_class_name = idx_to_class[0]

            # Ajout du résultat dans la liste
            predictions.append({
                'Id': img_id,
                'Class': predicted_class_name
            })

    # Sauvegarde dans un fichier CSV
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv(output_file, index=False)
    print(f"\nFichier de soumission généré avec succès : '{output_file}'")
    print("Vous pouvez maintenant l'envoyer sur Kaggle !")

