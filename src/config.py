import os

# Racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dossiers principaux
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Images
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test')

# CSV
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')

# Outputs
# On les sauvegarde à la racine du projet
MODEL_WEIGHTS = os.path.join(BASE_DIR, 'output/best_alexnet_model.pth')
SUBMISSION_FILE = os.path.join(BASE_DIR, 'output/submission.csv')

# Hyperparamètres
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
NUM_CLASSES = 315
