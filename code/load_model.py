import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class MyEnsemble(nn.Module):
    """
    Sequential model class with a skipped pathway.
    """

    def __init__(self, seg_model, clf_model):
        super(MyEnsemble, self).__init__()
        self.seg_model = seg_model
        self.clf_model = clf_model

    def forward(self, x):
        x1 = self.seg_model(x)
        x = torch.cat((x, x1), dim=1)
        assert x.shape[1] == 3
        x = self.clf_model(x)
        return x


def load_model(model_name, class_names):
    """
    Loads pre-trained models. 
    """
    # Load model
    ENCODER = "resnet50"
    ENCODER_WEIGHTS = "imagenet"
    CLASSES = class_names
    ACTIVATION = "sigmoid"

    if model_name == "unet":
        # create segmentation model with pretrained encoder
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            in_channels=1,
        )
    elif model_name == "unetpp":
        # create segmentation model with pretrained encoder
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            in_channels=1,
        )
    elif model_name == "combined":

        # Load model
        seg_ENCODER = "resnet50"
        seg_ENCODER_WEIGHTS = "imagenet"
        CLASSES = class_names
        seg_ACTIVATION = "sigmoid"

        clf_ENCODER = "timm-tf_efficientnet_lite4"
        clf_ENCODER_WEIGHTS = None  # Not pretrained model because input is no image
        CLASSES = class_names
        clf_ACTIVATION = "sigmoid"

        # create segmentation model with pretrained encoder
        seg_model = smp.Unet(
            encoder_name=seg_ENCODER,
            encoder_weights=seg_ENCODER_WEIGHTS,
            classes=2,  # Only 0 or 1 mask
            activation=seg_ACTIVATION,
            in_channels=1,
        )
        clf_model = smp.UnetPlusPlus(
            encoder_name=clf_ENCODER,
            encoder_weights=clf_ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=clf_ACTIVATION,
            in_channels=3,  # 2 for the segmentation channel output and one for the skipped connection
        )
        model = MyEnsemble(seg_model, clf_model)
    return model
