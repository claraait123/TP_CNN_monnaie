import torch
import torch.nn as nn
from torchvision import models

def get_alexnet_model(num_classes=315, pretrained=True):
    """
    Charge l'architecture AlexNet et modifie son classifieur final 
    pour s'adapter au nombre de classes du jeu de données (315 classes).
    """
    if pretrained:
        print("Chargement d'AlexNet avec les poids pré-entraînés sur ImageNet...")
        weights = models.AlexNet_Weights.DEFAULT
        model = models.alexnet(weights=weights)
    else:
        print("Chargement d'AlexNet sans poids pré-entraînés (entraînement from scratch)...")
        model = models.alexnet(weights=None)
  
    # On récupère la taille de l'entrée de cette dernière couche (4096 neurones)
    num_ftrs = model.classifier[6].in_features
    
    # On remplace cette couche par une nouvelle couche linéaire adaptée à notre problème
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    return model

