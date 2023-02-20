import os, cv2  # Read image as matrixes of shape (w, h, c)
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import imageio as iio

from skimage.io import imread, imsave  # Reads image as matrixes of shape (w, h)
import glob

pixel_value = {
    "stomach": [255, 0, 0],
    "large_bowel": [0, 255, 0],
    "small_bowel": [0, 0, 255],
}


def list_pixels_to_mask(folder_path, annot_path, plot_mask=False):
    """
    Save masks with four different pixel values based on the class of the pixel: None, stomach, large bowel, small bowel.
    Input:
        folder_path: path of the folder that contains the scan images.
        annot_path: path of the csv file containing the pixel lists of the organs contain in the scans. 
    """
    all_pngs = glob.glob(os.path.join(folder_path, "*.png"))
    annotations = pd.read_csv(annot_path)
    img_paths = [elt for elt in all_pngs if "img" in elt]
    for img_path in tqdm(img_paths):
        # Fetch image to get shape
        img = iio.imread(img_path)
        w, h = img.shape
        # Initiate mask to [0, 0, 0]
        img_mask = np.zeros((w, h, 3))
        # Get annotations
        annot_id = img_path.split("\\")[-1].split("_img")[0]
        annot = annotations[annotations.id == annot_id]
        annot.dropna(inplace=True)
        # Change 0 pixel value to class pixel value for every organ
        for organ in annot["class"].unique():
            organ_annot = annot[annot["class"] == organ]
            color = pixel_value[organ]
            pixels = organ_annot["segmentation"].values[0].split(" ")
            for i in range(0, len(pixels) - 1, 2):
                pix1, nbr_pix = int(pixels[i]), int(pixels[i + 1])
                i, j = pix1 // h, pix1 % h
                img_mask[i - 1, j - 1 : j + nbr_pix] = color
        # Save the mask image with similar file name as corresponding image
        mask_path = img_path.replace("img", "mask")
        img_mask = img_mask.astype(np.uint8)
        os.remove(mask_path)
        imsave(mask_path, img_mask)
        if plot_mask:
            img_mask = imread(mask_path)
            _, axes = plt.subplots(1, 2, figsize=(10, 8))
            axes[0].imshow(img, cmap="gray")
            axes[1].imshow(img_mask)
            plt.show()


list_pixels_to_mask("train\\**\\**\\scans\\**\\", "train.csv")
