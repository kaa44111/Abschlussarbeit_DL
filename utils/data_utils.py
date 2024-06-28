import torch
from torchvision.transforms import v2
import os
import shutil
from sklearn.model_selection import train_test_split

# Define the label mapping
MAPPING = {
    'form_0': 1,
    'form_1': 2,
    'form_2': 3,
    'form_3': 4,
    'form_4': 5,
    'form_5': 6
}

class BinningTransform(torch.nn.Module):
    def __init__(self, bin_size):
        super().__init__()
        self.bin_size = bin_size

    def forward(self, img):
        if img.dtype != torch.float32:
            img = img.float()
        C, H, W = img.shape
        new_H = H // self.bin_size
        new_W = W // self.bin_size
        img_binned = img.view(C, new_H, self.bin_size, new_W, self.bin_size).mean(dim=(2, 4))
        return img_binned
    

class PatchTransform(torch.nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, img):
        C, H, W = img.shape
        patches = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(1, 0, 2, 3)  # [num_patches, C, patch_size, patch_size]
        return patches

    
def split_data(root_dir, train_dir, val_dir, test_size=0.2, random_state=42):
    """
    Split Data into test and validate folders
    Example:
        root_dir = 'data/geometry_shapes'
        train_dir = 'data/circle_data/train'
        val_dir = 'data/circle_data/val'
    """
    # Erstellen der Zielordner
    for dir in [train_dir, val_dir]:
        grabs_dir = os.path.join(dir, 'grabs')
        masks_dir = os.path.join(dir, 'masks')
        os.makedirs(grabs_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
    
    # Pfade zu den Bilder- und Maskenordnern
    image_folder = os.path.join(root_dir, 'grabs')
    mask_folder = os.path.join(root_dir, 'masks')
    
    # Listen der Bild- und Maskendateien
    image_files = sorted(os.listdir(image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
    mask_files = sorted(os.listdir(mask_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    # Aufteilen der Daten in Trainings- und Validierungssätze
    train_images, val_images = train_test_split(image_files, test_size=test_size, random_state=random_state, shuffle=False)

    # Kopieren der Trainingsdaten
    for image_file in train_images:
        shutil.copy(os.path.join(image_folder, image_file), os.path.join(train_dir, 'grabs', image_file))
        # Annahme, dass die Maske denselben Namen wie das Bild hat, jedoch mit einer Endung von 1
        mask_file = image_file.split('.')[0] + '1.png'
        if os.path.exists(os.path.join(mask_folder, mask_file)):
            shutil.copy(os.path.join(mask_folder, mask_file), os.path.join(train_dir, 'masks', mask_file))

    # Kopieren der Validierungsdaten
    for image_file in val_images:
        shutil.copy(os.path.join(image_folder, image_file), os.path.join(val_dir, 'grabs', image_file))
        # Annahme, dass die Maske denselben Namen wie das Bild hat, jedoch mit einer Endung von 1
        mask_file = image_file.split('.')[0] + '1.png'
        if os.path.exists(os.path.join(mask_folder, mask_file)):
            shutil.copy(os.path.join(mask_folder, mask_file), os.path.join(val_dir, 'masks', mask_file))


def rename_masks(mask_folder,image_folder):
    '''
    Gives Masks the same prefix as the Image name
    '''
    image_files = sorted(os.listdir(image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
    mask_files = sorted(os.listdir(mask_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))

    for i, image_file in enumerate(image_files):
        base_name = image_file.split('.')[0]
        for j in range(6):  # Annahme: Es gibt 6 Masken pro Bild
            old_mask_name = mask_files[i * 6 + j]
            new_mask_name = f"{base_name}{j}.png"
            old_mask_path = os.path.join(mask_folder, old_mask_name)
            new_mask_path = os.path.join(mask_folder, new_mask_name)
            
            if os.path.exists(old_mask_path):
                os.rename(old_mask_path, new_mask_path)
                print(f"Renamed {old_mask_name} to {new_mask_name}")

