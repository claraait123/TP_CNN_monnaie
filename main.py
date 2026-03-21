import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


from src.config import *
from src.data_exploration import explore_data
from src.train import train_model
from src.inference import generate_submission

def check_data_folders():
    """Vérifie si les dossiers de données requis existent avant de lancer les scripts."""
    if not os.path.exists(TRAIN_IMG_DIR) or not os.path.exists(TRAIN_CSV):
        print(f"\nERREUR : Les données d'entraînement {TRAIN_IMG_DIR} et {TRAIN_CSV} sont introuvables.")
        return False
    return True

def main():
    if check_data_folders():
                print("\n----- Exploration -----")
                explore_data(csv_path=TRAIN_CSV)
                
                print("\n----- Entraînement -----")
                train_model(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
                
                if os.path.exists(MODEL_WEIGHTS) and os.path.exists(TEST_IMG_DIR):
                    print("\n----- Inférence -----")
                    generate_submission(
                        model_path=MODEL_WEIGHTS,
                        test_csv=TEST_CSV,
                        test_dir=TEST_IMG_DIR,
                        output_file=SUBMISSION_FILE
                    )

if __name__ == "__main__":
    # Point d'entrée principal
    main()