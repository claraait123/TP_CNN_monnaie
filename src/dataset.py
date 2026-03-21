import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

from config import TRAIN_CSV, TRAIN_IMG_DIR, BATCH_SIZE

class CoinDataset(Dataset):
    """
    Classe Dataset PyTorch personnalisée pour le chargement des images de pièces.
    """
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = str(self.dataframe.loc[idx, 'Id'])
        label = self.dataframe.loc[idx, 'Label']
        
        search_pattern = os.path.join(self.img_dir, f"{img_id}.*")
        matching_files = glob.glob(search_pattern)
        
        if not matching_files:
            print(f"Avertissement : Aucune image trouvée pour l'ID {img_id} dans {self.img_dir}")
            # Image noire par défaut si introuvable
            image = Image.new("RGB", (227, 227), (0, 0, 0))
        else:
            img_path = matching_files[0]
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Erreur de chargement pour l'image {img_path}: {e}")
                image = Image.new("RGB", (227, 227), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders(csv_path=TRAIN_CSV, img_dir=TRAIN_IMG_DIR, batch_size=BATCH_SIZE, test_size=0.2):
    """
    Prépare les données, encode les labels, et retourne les DataLoaders PyTorch.
    """
    # Chargement du CSV
    df = pd.read_csv(csv_path)
    
    # Création du dictionnaire de classes
    unique_classes = sorted(df['Class'].unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    
    # Ajout de la colonne des labels entiers
    df['Label'] = df['Class'].map(class_to_idx)
    
    # Séparation Train/Validation
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42, 
        stratify=df['Label']
    )
    
    # Définition des transformations (AlexNet : 227x227)
    train_transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation
    val_transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Instanciation
    train_dataset = CoinDataset(train_df, img_dir, transform=train_transforms)
    val_dataset = CoinDataset(val_df, img_dir, transform=val_transforms)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Nombre d'images d'entraînement : {len(train_dataset)}")
    print(f"Nombre d'images de validation : {len(val_dataset)}")
    print(f"Nombre de classes : {len(unique_classes)}")
    
    return train_loader, val_loader, idx_to_class
