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

def custom_collate_fn(batch):
    """
    Only for Datasets with more that 1 Feature
    Custom collate function for DataLoader.
    This function assumes each item in batch is a tuple (image, masks, combined_mask).
    """

    # Filter out any None values (in case of any loading errors)
    batch = [item for item in batch if item[0] is not None]

    images, masks, combined_masks = zip(*batch)
    batched_images = torch.stack(images)
    batched_combined_masks = torch.stack(combined_masks)
    batched_masks = torch.stack(masks)
    
    return batched_images, batched_masks, batched_combined_masks

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
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
        
    image_folder = os.path.join(root_dir, 'grabs')
    mask_folder = os.path.join(root_dir, 'masks')
    
    image_files = sorted(os.listdir(image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
    mask_files = sorted(os.listdir(mask_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_files, mask_files, test_size=test_size, random_state=random_state)

    for image_file in train_images:
        shutil.copy(os.path.join(image_folder, image_file), os.path.join(train_dir, 'grabs', image_file))
    for mask_file in train_masks:
        shutil.copy(os.path.join(mask_folder, mask_file), os.path.join(train_dir, 'masks', mask_file))
        
    for image_file in val_images:
        shutil.copy(os.path.join(image_folder, image_file), os.path.join(val_dir, 'grabs', image_file))
    for mask_file in val_masks:
        shutil.copy(os.path.join(mask_folder, mask_file), os.path.join(val_dir, 'masks', mask_file))