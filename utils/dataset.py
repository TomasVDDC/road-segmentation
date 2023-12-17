import torch
import os
import cv2
import numpy as np
import albumentations as albu
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
  CLASSES = ['road']

  def __init__(self, images_dir, masks_dir=None, classes=None, augmentation=None, preprocessing=None, plot = False):
      if masks_dir == None:
        self.ids = range(1, 51)
        self.images_path = [os.path.join(images_dir, f'test_{idx}/',f'test_{idx}.png') for idx in self.ids]
      else:
        self.ids = os.listdir(images_dir)
        self.images_path = [os.path.join(images_dir, image_id) for image_id in self.ids]
      self.masks_path = [os.path.join(masks_dir, image_id) for image_id in self.ids] if masks_dir is not None else None

      # convert str names to class values on masks
      if classes is not None:
          self.class_values = [self.CLASSES.index(cls.lower())*255 for cls in classes]

      self.augmentation = augmentation ###plus besoin ?
      self.preprocessing = preprocessing
      # self.preprocessing = None
      self.plot = plot

  def __getitem__(self, i):

      # read data
      image = cv2.imread(self.images_path[i])
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      height, width, channel = image.shape
      if (height % 32) or (width % 32):
          image = cv2.resize(image, (416, 416)) ###

      # initialize mask as None
      mask = None

      if self.masks_path == None:
        if self.augmentation: ###plus besoin ?
          sample = self.augmentation(image=image)
          image = sample['image']
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
        return self.images_path[i], image

      else:
        mask = cv2.imread(self.masks_path[i], 0)
        if (height % 32) or (width % 32): ###
            mask = cv2.resize(mask, (416, 416))
            mask[mask<=120] = 0 #pixel value {0, 255}
            mask[mask>120] = 255
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
              sample = self.preprocessing(image=image, mask=mask)
              image, mask = sample['image'], sample['mask']
        return image, mask

  def __len__(self):
      return len(self.ids)
  
  

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)