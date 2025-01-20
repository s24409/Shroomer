import cv2
import os
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2


def get_transforms(
    model_mean: list[float], model_std: list[float], image_size: tuple[int, int]
) -> Compose:
    """
    Get the transformations to be applied to the images.

    Parameters
    ----------
    model_mean : list of float
        Mean values for normalization.
    model_std : list of float
        Standard deviation values for normalization.
    image_size : tuple of int
        Desired image size (height, width).

    Returns
    -------
    albumentations.core.composition.Compose
        Composition of transformations.
    """
    return Compose(
        [Resize(*image_size), Normalize(mean=model_mean, std=model_std), ToTensorV2()]
    )


def preprocess_image(image_path):
    """Preprocess image for model input using the same transformations as training"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Failed to read image")
        
        # Convert BGR to RGB and ensure it's a numpy array
        img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Ensure correct dtype
        img = img.astype(np.uint8)
            
        model_mean = [0.5, 0.5, 0.5]
        model_std = [0.5, 0.5, 0.5]
        transforms = get_transforms(model_mean, model_std, (224, 224))
        
        # Apply transformations
        augmented = transforms(image=img)
        tensor_img = augmented["image"]
        
        return tensor_img
        
    except Exception as e:
        print(f"Error details: {str(e)}")
        raise Exception(f"Error preprocessing image: {str(e)}")
