# TP : Classification de pièces de monnaie (CNN)

Ce dépôt contient le code source pour le TP "Classification de pièces" du module **Réseaux de neurones pour la vision par ordinateur** (Master 1 Informatique VMI).

## Objectif

L'objectif de ce projet est de construire un modèle de vision par ordinateur capable d'identifier la valeur, la devise et le pays d'origine d'une pièce de monnaie à partir d'une simple image, parmi **315 classes distinctes**.

## Contexte

Ce projet utilise une architecture de **réseau de neurones convolutifs (CNN)** basée sur **AlexNet** pour effectuer une classification multi-classe sur un ensemble de données de monnaies. Le modèle doit être capable de généraliser à partir des images d'entraînement pour prédire correctement les classes des images de test.

---

## Dataset

Le dataset est organisé de la manière suivante :

```
data/
├── train/               
│   ├── 1.jpg          
│   ├── 2.jpg
│   ├── 3.jpg
│   └── ...
├── test/               # Images de test sans labels
├── train.csv           # Labels d'entraînement (Id, Label)
├── test.csv            # Ids de test
└── sample_submission.csv # Format attendu pour les prédictions
```

## Installer les dépendances

```bash
pip install -r requirements.txt
```

## Utilisation

### Exécution complète du pipeline

Pour lancer l'ensemble du pipeline (exploration → entraînement → inférence) :

```bash
python main.py
```