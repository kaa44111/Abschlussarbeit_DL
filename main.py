import sys
import os

# Den Projektpfad zu sys.path hinzufügen
project_path = os.path.abspath(os.path.dirname(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Import using different methods
from models.model import UNet
from models.test_model import test


if __name__ == '__main__':
    try:
        test(UNet=UNet)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")