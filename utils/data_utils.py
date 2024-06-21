import torch
from torchvision.transforms import v2


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

# def custom_collate_fn(batch):
#     """
#     Custom collate function for DataLoader.
#     This function assumes each item in batch is a tuple (image, combined_mask, masks).
#     """

#     # Filter out any None values (in case of any loading errors)
#     batch = [item for item in batch if item[0] is not None]

#     images, combined_masks = zip(*batch)
#     batched_images = torch.stack(images)
#     batched_combined_masks = torch.stack(combined_masks)
    
#     return batched_images, batched_combined_masks

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