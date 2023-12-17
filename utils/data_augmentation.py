# -*- coding: utf-8 -*-
"""ML data augmentation project 2 - road segmentation"""

import os
from os import listdir
import numpy as np
import cv2
import albumentations as albu
from ipywidgets import widgets
from IPython.display import display, clear_output



def confirm_and_augment():

    def on_button_clicked(b):
        if b.description == 'Yes':
            clear_output(wait=True)
            print("Proceeding with data augmentation...")
            augment_data()
        elif b.description == 'No':
            clear_output(wait=True)
            print("Data augmentation canceled.")

    yes_button = widgets.Button(description="Yes")
    no_button = widgets.Button(description="No")

    yes_button.on_click(on_button_clicked)
    no_button.on_click(on_button_clicked)

    display(widgets.HBox([yes_button, no_button]))
    print("Do you want to proceed with data augmentation? If augmented data already exists press <No>")


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created")
    else:
        print(f"Directory {path} already exists")

# Create folders for data augmentation
def augment_data():
    create_directory("data/data_train_augmented") 
    create_directory("data/data_train_augmented/images/") 
    create_directory("data/data_train_augmented/masks/") 
    create_directory("data/data_train_augmented/raw/") 
    create_directory("data/data_train_augmented/raw/images/") 
    create_directory("data/data_train_augmented/raw/masks/") 
    create_directory("data/data_validation") 
    create_directory("data/data_validation/images/") 
    create_directory("data/data_validation/masks/") 
    create_directory("data/data_validation/raw/")
    create_directory("data/data_validation/raw/images/") 
    create_directory("data/data_validation/raw/masks/") 

    # Load images and masks from dataset
    PATH_IMG_TRAIN = "./data/training/images/"
    PATH_MASK_TRAIN = "./data/training/groundtruth/"
    img_train, mask_train = load_img_training(PATH_IMG_TRAIN, PATH_MASK_TRAIN)
    key_list = list(img_train.keys())
    key_list.sort()

    # Split the images for training/validation (+ store)
    training_ratio = 0.80
    seed = 1
    train_keys, val_keys = split_keys(np.array(key_list), training_ratio=training_ratio, seed=seed)

    PATH_TR_IMG_AUG_RAW = "./data/data_train_augmented/raw/images/"
    PATH_TR_MASK_AUG_RAW = "./data/data_train_augmented/raw/masks/"
    PATH_VAL_IMG_RAW = "./data/data_validation/raw/images/"
    PATH_VAL_MASK_RAW = "./data/data_validation/raw/masks/"

    store_images(img_train, train_keys, PATH_TR_IMG_AUG_RAW)
    store_images(mask_train, train_keys, PATH_TR_MASK_AUG_RAW)
    store_images(img_train, val_keys, PATH_VAL_IMG_RAW)
    store_images(mask_train, val_keys, PATH_VAL_MASK_RAW)

    MASK_THRESHOLD = 120
    SIZE_X = 416 #divisible by 32
    SIZE_Y = 416 #divisible by 32
    PATH_TR_IMG_AUG = "./data/data_train_augmented/images/"
    PATH_TR_MASK_AUG = "./data/data_train_augmented/masks/"
    PATH_VAL_IMG = "./data/data_validation/images/"
    PATH_VAL_MASK = "./data/data_validation/masks/"

    # Load validation images and resize
    img_val_raw, mask_val_raw = load_img_training(PATH_VAL_IMG_RAW, PATH_VAL_MASK_RAW)
    keys_val = list(img_val_raw.keys())
    resize_augment_store_dataset(img_val_raw, mask_val_raw, keys_val, SIZE_Y, SIZE_X, MASK_THRESHOLD, PATH_VAL_IMG, PATH_VAL_MASK, augment=False)

    # Load training images, resize and augment using geometric transformation (+ store)
    img_tr_raw, mask_tr_raw = load_img_training(PATH_TR_IMG_AUG_RAW, PATH_TR_MASK_AUG_RAW)
    keys_tr = list(img_tr_raw.keys())
    resize_augment_store_dataset(img_tr_raw, mask_tr_raw, keys_tr, SIZE_Y, SIZE_X, MASK_THRESHOLD, PATH_TR_IMG_AUG, PATH_TR_MASK_AUG, augment=True)
    print("Data augmentation completed.")



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

def load_img_training(path_img, path_mask):
    """
      Load training images and corresponding masks from specified directories.

      Args:
      - path_img (str): The path to the directory containing training images. Images should be in PNG format.
      - path_mask (str): The path to the directory containing corresponding masks for training images. Masks should be in PNG format and grayscale.

      Returns:
      - dict: A dictionary containing training images loaded from the specified directory.
      - dict: A dictionary containing corresponding masks for the training images loaded from the specified directory.
    """
    train_img = {}
    images = listdir(path_img)
    images = [img for img in images if img.endswith(".png")]

    for image in images:
        img = cv2.imread(path_img + image, cv2.IMREAD_COLOR)
        train_img[image] = img

    train_mask = {}
    masks = listdir(path_mask)
    masks = [img for img in masks if img.endswith(".png")]

    for mask in masks:
        img = cv2.imread(path_mask + mask, cv2.IMREAD_GRAYSCALE)
        train_mask[mask] = img

    return train_img, train_mask

  