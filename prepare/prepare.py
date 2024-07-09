from numpy.lib.stride_tricks import as_strided
from PIL import Image
import numpy as np
import cv2

def histogram_equalization_cv2(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    return img_equalized

def downsample_image(input_path, output_path, scale_factor):
    # Bild laden
    img = Image.open(input_path)
    
    # Neue Größe berechnen
    new_size = (int(img.width / scale_factor), int(img.height / scale_factor))
    
    # Bildgröße ändern (runterskalieren)
    downsampled_img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Verkleinertes Bild speichern
    downsampled_img.save(output_path)



if __name__ == '__main__':
    try:
        # Load images
        scale_factor = 4  # Verkleinerungsfaktor (z.B. auf 1/4 der ursprünglichen Größe)
        image_path = 'data/WireCheck/grabs/00Grab (2).tiff'
        mask_path = 'data/WireCheck/masks/00Grab (3).tif'


        output_image_path = 'prepare/test/downsample_function(3).tiff'
        print('Downsampling function:')
        downsample_image(image_path, output_image_path, scale_factor)


    except Exception as e:
        print(f"An error occurred: {e}")

