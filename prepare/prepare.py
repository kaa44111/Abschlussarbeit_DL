from numpy.lib.stride_tricks import as_strided
from PIL import Image
import numpy as np


def bin_image(input_path, output_path, bin_size):
    # Bild laden und in ein numpy-Array konvertieren
    img = Image.open(input_path)
    img_array = np.array(img)
    
    # Prüfen, ob das Bild mehr als eine Farbkanal hat
    if len(img_array.shape) == 2:  # Graustufenbild
        img_array = img_array[:, :, np.newaxis]

    # Neue Größe berechnen
    new_shape = (img_array.shape[0] // bin_size, bin_size,
                 img_array.shape[1] // bin_size, bin_size, img_array.shape[2])
    
    # Binning durchführen
    binned_img_array = img_array.reshape(new_shape).mean(axis=(1, 3))
    
    # Binned Bild konvertieren und speichern
    if binned_img_array.shape[2] == 1:  # Graustufenbild
        binned_img_array = binned_img_array[:, :, 0]
    binned_img = Image.fromarray(binned_img_array.astype(img_array.dtype))
    binned_img.save(output_path)



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
        mask_path = 'data/WireCheck/masks/00Grab (2).tif'
        output_image_path = 'prepare/test/test.tiff'
        print('Downsampling function:')
        
        downsample_image(image_path, output_image_path, scale_factor)
        # img1 = Image.open(output_image_path)
        # img1 = np.asarray(img1)
        # print(img1.shape)

        print('Bin function:')
        output_path = 'prepare/test/test1.tiff'
        bin_image(image_path, output_path, 4)
        img2 = Image.open(output_path)
        img2 = np.asarray(img2)
        print(img2.shape)

    except Exception as e:
        print(f"An error occurred: {e}")

