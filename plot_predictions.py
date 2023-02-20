import os, cv2
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import glob

from skimage.io import imread
from preprocess.one_hot_encoding import reverse_one_hot
from train import mask_preprocess, img_preprocess, DataLoaderSegmentation


def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20, 5))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace("_", " ").title())  # fontsize=20
        if "ruthd" in name:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)
    plt.savefig("images\\1.png", bbox_inches="tight", pad_inches=0.1)
    plt.show()


# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def vizualize_prediction(img, label, model, class_rgb_values, device, save_path=None):
    """
    img and label must be object from dataset (DataLoaderSegmentation type).
    """
    model.eval()
    with torch.no_grad():
        img.to(device)
        img_tensor = img.unsqueeze(0)
        pred = model(img_tensor)
    label = label.permute(1, 2, 0)
    label = label.numpy()
    pred = pred.squeeze(0)
    pred = pred.permute(1, 2, 0)
    pred = pred.numpy()
    visualize(
        original_image=np.squeeze(img),
        ground_truth_mask=colour_code_segmentation(
            reverse_one_hot(label), class_rgb_values
        ).astype("float"),
        predicted_mask=colour_code_segmentation(
            reverse_one_hot(pred), class_rgb_values
        ).astype("float"),
    )


def load_trained_model(model, model_path):
    # Set device: `cuda` or `cpu`
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.double()
    print(f"{model.name} is loaded")
    return model


def get_validation_ds(select_class_rgb_values, test_folder="test\\**\\**\\scans\\**\\"):
    """
    Load validation dataset to visualise models prediction. 
    """
    valid_dataset = DataLoaderSegmentation(
        test_folder,
        img_preprocessing=img_preprocess,
        mask_preprocessing=mask_preprocess,
        class_rgb_values=select_class_rgb_values,
    )
    return valid_dataset
