# -*- coding: utf-8 -*-
"""ML data augmentation project 2 - road segmentation"""

import os
from os import listdir
import cv2
import numpy as np
import albumentations as albu

def resize_augment_store_dataset(img_dict, mask_dict, keys, size_y, size_x, mask_threshold, path_img, path_mask, augment=False):
    """
    Resize, augment, and store a dataset of images and masks.

    Args:
    - img_dict (dict): Dictionary containing original images with keys.
    - mask_dict (dict): Dictionary containing corresponding masks with keys.
    - keys (list): List of keys representing the images and masks to be processed.
    - size_y (int): Target height of the resized images and masks.
    - size_x (int): Target width of the resized images and masks.
    - mask_threshold (int): Threshold value for binarizing masks.
    - path_img (str): Path to store the resized and augmented images.
    - path_mask (str): Path to store the resized and augmented masks.
    - augment (bool, optional): Flag indicating whether to perform data augmentation. Default is False.

    Returns:
    - None
    """
    img_resized = {}
    mask_resized = {}
    resized = "r_"
    for key in keys:
    img = cv2.resize(img_dict[key], (size_y, size_x))
    img_resized[resized+key] = img
    mask = cv2.resize(mask_dict[key], (size_y, size_x))
    mask[mask<mask_threshold] = 0 #pixel value {0, 255}
    mask[mask>mask_threshold] = 255
    mask_resized[resized+key] = mask

    if augment:
    # Data augmentation
    # Flip: horizontal and vertical
    h_flipped = "h_"
    v_flipped = "v_"
    aug_h = albu.HorizontalFlip(p=1)
    aug_v = albu.VerticalFlip(p=1)

    key_resized = list(img_resized.keys())
    for key in key_resized:
        h = aug_h(image=img_resized[key], mask=mask_resized[key])
        v = aug_v(image=img_resized[key], mask=mask_resized[key])
        img_resized[h_flipped+key] = h['image']
        mask_resized[h_flipped+key] = h['mask']
        img_resized[v_flipped+key] = v['image']
        mask_resized[v_flipped+key] = v['mask']

    # Rotation: 90°, 180°, 270°
    rot_90 = "90_"
    rot_180 = "180_"
    rot_270 = "270_"
    key_resized_flipped = list(img_resized.keys())
    for key in key_resized_flipped:
        rot90_img = np.rot90(img_resized[key])
        img_resized[rot_90+key] = rot90_img
        rot180_img = np.rot90(rot90_img)
        img_resized[rot_180+key] = rot180_img
        rot270_img = np.rot90(rot180_img)
        img_resized[rot_270+key] = rot270_img

        rot90_mask = np.rot90(mask_resized[key])
        mask_resized[rot_90+key] = rot90_mask
        rot180_mask = np.rot90(rot90_mask)
        mask_resized[rot_180+key] = rot180_mask
        rot270_mask = np.rot90(rot180_mask)
        mask_resized[rot_270+key] = rot270_mask

    store_images(img_resized, list(img_resized.keys()), path_img)
    store_images(mask_resized, list(mask_resized.keys()), path_mask)

def store_images(img_dict, keys, output_path):
    """
    Store images from a dictionary in a specified directory with filenames represented by keys.

    Args:
    - img_dict (dict): Dictionary containing images with keys.
    - keys (list): List of keys representing the images to be stored.
    - output_path (str): The path to the directory where the images will be saved.

    Returns:
    - None
    """
    for key in keys:
        # Full path to save the image
        save_path = os.path.join(output_path, key)
        cv2.imwrite(save_path, img_dict[key])

    print(f"Images stored in {output_path}")
    
def split_keys(keys, training_ratio=0.8, seed=1):
    """
    Split a list of keys into training and validation sets.

    Args:
    - keys (numpy.ndarray): Array containing keys to be split.
    - training_ratio (float, optional): The ratio of keys to be used for training. Default is 0.8 (80%).
    - seed (int, optional): Seed for reproducibility. Default is 1.

    Returns:
    - numpy.ndarray: Array containing keys for the training set.
    - numpy.ndarray: Array containing keys for the validation set.
    """
    np.random.seed(seed)
    num_keys = len(keys)

    # Create a random permutation of indices
    indices = np.random.permutation(num_keys)

    # Calculate the number of samples for the training set
    num_training = int(training_ratio * num_keys)

    # Split images and masks into training and validation sets
    train_indices = indices[:num_training]
    val_indices = indices[num_training:]
    train_keys, val_keys = keys[train_indices], keys[val_indices]
    return np.array(train_keys), np.array(val_keys)

  