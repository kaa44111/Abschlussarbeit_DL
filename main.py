import sys
import os

# Den Projektpfad zu sys.path hinzufügen
project_path = os.path.abspath(os.path.dirname(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Import using different methods
from models.model import UNet
from models.test_model import test
from utils.data_utils import split_data

if __name__ == '__main__':
    try:
        root_dir = 'data/geometry_shapes'
        train_dir = 'data/circle_data/train'
        val_dir = 'data/circle_data/val'
    
        split_data(root_dir, train_dir, val_dir, test_size=0.2, random_state=42)
        print("Data split completed.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")