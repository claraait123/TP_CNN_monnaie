import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os

from config import TRAIN_CSV, TRAIN_IMG_DIR, MODEL_WEIGHTS, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE

# Importation de vos modules locaux
from src.dataset import get_dataloaders
from src.model import get_alexnet_model

def train_model(num_epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement lancé sur : {device}")

    # Préparation des données
    print("Préparation des DataLoaders...")
    train_loader, val_loader, idx_to_class = get_dataloaders(
        csv_path=TRAIN_CSV, 
        img_dir=TRAIN_IMG_DIR, 
        batch_size=BATCH_SIZE
    )

    # Initialisation du modèle
    model = get_alexnet_model(num_classes=315, pretrained=True)
    model = model.to(device)

    # Fonction de perte et Optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Historique
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    # 5. Boucle d'apprentissage
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nÉpoque {epoch+1}/{num_epochs}")
        print("-" * 20)

        # --- PHASE D'ENTRAÎNEMENT ---
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # Réinitialise les gradients

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            # Statistiques
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)

        # --- PHASE DE VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_corrects.double() / len(val_loader.dataset)

        # Sauvegarde de l'historique
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc.item())
        history['val_acc'].append(epoch_val_acc.item())

        epoch_time = time.time() - start_time
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")
        print(f"Temps: {epoch_time:.0f}s")

        # Sauvegarde du meilleur modèle
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_WEIGHTS)

    print(f"\nEntraînement terminé. Meilleure précision de validation : {best_val_acc:.4f}")
    
    # Affichage et sauvegarde des courbes d'apprentissage
    plot_learning_curves(history, num_epochs)

def plot_learning_curves(history, num_epochs):
    """Génère le graphique des pertes et précisions (Loss & Accuracy)."""
    epochs_range = range(1, num_epochs + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Courbe de la perte (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.title('Évolution de la Perte (Loss)')
    plt.xlabel('Époques')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Courbe de la précision (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.title('Évolution de la Précision')
    plt.xlabel('Époques')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('output/learning_curves.png', dpi=300)
    print("\nGraphique des courbes sauvegardé sous 'output/learning_curves.png'")
    plt.show()

