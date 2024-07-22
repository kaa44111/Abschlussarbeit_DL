import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.utils import draw_segmentation_masks
from torchvision.ops import masks_to_boxes
from utils.data_utils import find_image_and_mask_files_folder

def pad_image(image, target_size):
    """Pad the image to the target size."""
    width, height = image.size
    new_image = Image.new("RGB" if image.mode == "RGB" else "L", target_size)
    new_image.paste(image, ((target_size[0] - width) // 2, (target_size[1] - height) // 2))
    return new_image

def process_and_save_cropped_images(root_dir, dataset_name, target_size):
    # Find image and mask files
    image_folder, mask_folder, image_files, mask_files = find_image_and_mask_files_folder(root_dir, dataset_name)
    
    # Create output directories
    output_dir = os.path.join('data_modified', dataset_name, 'bounding_box')
    output_image_folder = os.path.join(output_dir, 'grabs')
    output_mask_folder = os.path.join(output_dir, 'bounding_box', 'masks')
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    # Process each image and mask
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(mask_folder, mask_file)

        # Load image and mask
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Ensure mask has the same size as the image
        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)

        # Convert mask to tensor and find bounding boxes
        mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).float()
        masks = mask_tensor > 0
        bool_masks = masks.byte()
        boxes = masks_to_boxes(bool_masks)

        # Crop and save images based on bounding boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.tolist())
            cropped_img = img.crop((x1, y1, x2, y2))
            cropped_mask = mask.crop((x1, y1, x2, y2))

            # Pad the cropped images to the target size
            padded_img = pad_image(cropped_img, target_size)
            padded_mask = pad_image(cropped_mask, target_size)

            padded_img.save(os.path.join(output_image_folder, f'{os.path.splitext(img_file)[0]}_crop_{i}.tif'))
            padded_mask.save(os.path.join(output_mask_folder, f'{os.path.splitext(mask_file)[0]}_crop_{i}.tif'))

    return output_dir

# Beispielaufruf der Funktion
root_dir = 'data/Dichtflächen'
dataset_name = 'Dichtflächen'
# test = process_and_save_cropped_images(root_dir, dataset_name,target_size=(160, 160))
# a=1

# # Root directory for the images and masks
# root_dir = 'data/Dichtflächen'

# plt.rcParams["savefig.bbox"] = "tight"

# def show(imgs):
#     if not isinstance(imgs, list):
#         imgs = [imgs]
#     fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
#     for i, img in enumerate(imgs):
#         img = img.detach()
#         img = F.to_pil_image(img)
#         axs[0, i].imshow(np.asarray(img))
#         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

# # Paths to the image and mask
# img_path = os.path.join(root_dir, 'grabs', '1.bmp')
# mask_path = os.path.join(root_dir, 'masks', '1.tif')

# # Load the image and mask
# img = Image.open(img_path).convert('RGB')
# mask = Image.open(mask_path).convert('L')

# # Ensure the mask has the same size as the image
# if img.size != mask.size:
#     mask = mask.resize(img.size, Image.NEAREST)

# # Convert the image and mask to tensors
# img = torch.from_numpy(np.array(img)).float() / 255.0
# mask = torch.from_numpy(np.array(mask)).float()

# # Remove batch dimension to the image and mask
# img = img.permute(2, 0, 1)  # Change from HWC to CHW
# mask = mask.unsqueeze(0)    # Add channel dimension to mask

# print("Image size:", img.size())
# print("Mask size:", mask.size())

# # Get the unique colors (object ids) from the mask
# obj_ids = torch.unique(mask)
# print("Object IDs:", obj_ids)

# # Remove the background (first id)
# obj_ids = obj_ids[1:]
# print("Foreground Object IDs:", obj_ids)

# # Split the mask into a set of boolean masks
# masks = mask == obj_ids[:, None, None]
# print("Masks size:", masks.size())

# # Convert image to uint8
# img_uint8 = (img * 255).byte()

# # Draw segmentation masks on the image
# drawn_masks = []
# for single_mask in masks:
#     drawn_img = draw_segmentation_masks(img_uint8, single_mask.bool(), alpha=0.8, colors="blue")
#     drawn_masks.append(drawn_img)

# # Show the images with drawn masks
# show(drawn_masks)


# # Convert masks to boxes
# bool_masks = masks.squeeze(1).byte()  # Ensure masks are [N, H, W]
# boxes = 
# print("Bounding boxes size:", boxes.size())
# print("Bounding boxes:", boxes)

# from torchvision.utils import draw_bounding_boxes

# drawn_boxes = draw_bounding_boxes(img_uint8, boxes, colors="red", width=3)
# show(drawn_boxes)
# plt.show()