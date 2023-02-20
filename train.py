import os, cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torchvision import transforms
from torchmetrics.functional import dice
import glob

from skimage.io import imread
import argparse

from load_model import load_model
from preprocess.one_hot_encoding import one_hot_encode, reverse_one_hot

parser = argparse.ArgumentParser(description="Train deep neural network.")
parser.add_argument(
    "--batch_size", metavar="BS", type=int, help="batch size for training"
)
parser.add_argument("--epochs", type=int, help="number of epoch for training")
parser.add_argument(
    "--model_name", type=str, help="model name could be either unet, unetpp or combined"
)


class DataLoaderSegmentation(torch.utils.data.Dataset):
    def __init__(
        self,
        folder_path,
        class_rgb_values=None,
        img_preprocessing=None,
        mask_preprocessing=None,
    ):
        super(DataLoaderSegmentation, self).__init__()
        all_pngs = glob.glob(os.path.join(folder_path, "*.png"))
        self.img_files = [elt for elt in all_pngs if "img" in elt]
        self.mask_files = [elt for elt in all_pngs if "mask" in elt]
        self.pixel_mask_value = class_rgb_values
        self.img_preprocessing = img_preprocessing
        self.mask_preprocessing = mask_preprocessing

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        data = imread(img_path) / 65535  # Because image pixel are 16bits
        label = imread(mask_path)

        label = one_hot_encode(label, self.pixel_mask_value).astype("float")

        # apply preprocessing
        if self.img_preprocessing:
            data = self.img_preprocessing(data)
        if self.mask_preprocessing:
            label = self.mask_preprocessing(label)

        return data, label

    def __len__(self):
        return len(self.img_files)


img_preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.CenterCrop((288, 288)),
        transforms.Normalize(mean=0.00164, std=0.00197),
    ]
)

mask_preprocess = transforms.Compose(
    [transforms.ToTensor(), transforms.CenterCrop((288, 288))]
)


def main(batch_size=16, epochs=2, model_name="unet"):
    class_dict = pd.read_csv("train.csv")
    # Get class names
    class_names = list(set(class_dict["class"].tolist()))
    class_names = ["no", "stomach", "large_bowel", "small_bowel"]

    select_classes = ["no", "stomach", "large_bowel", "small_bowel"]

    # Get RGB values of required classes
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    class_rgb_values = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    model = load_model(model_name, class_names)

    train_folder = "train/**/**/scans/**/"
    test_folder = "test/**/**/scans/**/"

    print("Get train and val dataset instances")
    train_dataset = DataLoaderSegmentation(
        train_folder,
        img_preprocessing=img_preprocess,
        mask_preprocessing=mask_preprocess,
        class_rgb_values=select_class_rgb_values,
    )

    valid_dataset = DataLoaderSegmentation(
        test_folder,
        img_preprocessing=img_preprocess,
        mask_preprocessing=mask_preprocess,
        class_rgb_values=select_class_rgb_values,
    )

    print("Number of images in train dataset", len(train_dataset))

    # Get train and val data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    # Set device: `cuda` or `cpu`
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    # define loss function
    dice_loss = smp.losses.DiceLoss(mode="multilabel", from_logits=False)
    # define optimizer
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.001),])

    # Training and Evaluation loop
    total_train_losses = []
    total_val = []
    train_losses = []
    val = []

    model = model.to(device)
    model.double()
    for epoch in range(1, epochs + 1):
        # TRAINING
        model.train()

        # Create progress bar
        i = 0
        trainloader = tqdm(train_loader, unit="batches", desc="Training: ...")
        for batch in trainloader:
            # Get images, labels and put them on appropriate device
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            # Get features output by the model
            features = model(imgs)
            # Call the loss
            loss = dice_loss(y_pred=features, y_true=labels)
            #######

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            train_loss_mean = np.mean(train_losses)
            trainloader.set_description(
                f'Training:\tepoch: {epoch}/{epochs}\tloss: {"{:.4f}".format(train_loss_mean)}'
            )
            i += 1

            if i % 100 == 0:
                print(f"Saving model for {i//200 + 1}th time")
                torch.save(
                    model.state_dict(), f"./models/unet_bs_{batch_size}_ep_{epochs}.pt"
                )
                # Saving train losses value for futur plots
                with open(
                    f"train_losses/unet_bs_{batch_size}_losses_{epoch}.txt", "w"
                ) as f:
                    for loss_value in train_losses:
                        f.write(f"{loss_value}\n")
        total_train_losses.append(train_loss_mean)

        # VALIDATION
        model.eval()
        valid_loader = tqdm(valid_loader, unit="batches", desc="Evaluating: ...")
        for batch in valid_loader:
            with torch.no_grad():
                # Get images, labels batch, put them to device
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)

                optimizer.zero_grad()
                # Get features output by the model
                features = model(imgs)
                # Compute average of the dice coefficient on the intestines classes only (None class is at index 0)
                val_metric = dice(features, labels, average="macro", ignore_index=0)
                ######

                val.append(val_metric.item())
                val_loss_mean = np.mean(val)
                valid_loader.set_description(
                    f'Validation:\tepoch: {epoch}/{epochs}\tloss: {"{:.4f}".format(val_loss_mean)}'
                )
        total_val.append(val_loss_mean)
        with open(
            f"train_losses/unet_val_bs_{batch_size}_losses_{epoch}.txt", "w"
        ) as f:
            for loss_value in val:
                f.write(f"{loss_value}\n")

    print(f"\n {i} images have been used for this training.")
    torch.save(model.state_dict(), f"./models/unet_bs_{batch_size}_ep_{epochs}.pt")


if __name__ == "__main__":
    args = parser.parse_args()
    batch_s, n_epoch, model_name = args.batch_size, args.epochs, args.model_name
    print(f"Training {model_name} for {n_epoch} epochs with batch size of {batch_s}")
    main(batch_s, n_epoch, model_name)
