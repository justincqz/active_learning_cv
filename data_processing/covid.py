from data_processing.dataprep import DataPrep
from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms as T

import os
import torch
import pandas as pd
import numpy as np

class CovidDataset(Dataset):
  """Covid dataset."""

  def __init__(self, df, root_dir, transform=None, shuffle=False, classes=None, train=False):
    """
    Args:
        df (Pandas.DataFrame): Pandas DataFrame containing a 'path' and 'label' column
        root_dir (string): Directory with stores the images
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.root = root_dir
    self.frame = df.copy()
    if shuffle:
      self.frame = self.frame.sample(frac= 1)
    self.root_dir = root_dir
    self.transform = transform
    self.classes = list(range(120)) if classes is None else classes
    self.data = np.array(self.frame.iloc[:, -2])
    self.targets = np.array(self.frame.iloc[:, -1])
    self.train = train

  def __len__(self):
    return len(self.frame)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    img_name = self.root_dir + self.frame.iloc[idx, -2]
    image = Image.open(img_name)
    if not image.mode == 'RGB':
      image = image.convert('RGB')
    label = int(self.frame.iloc[idx, -1])

    if self.transform:
      image = self.transform(image)

    return image, label

class CovidDS(DataPrep):
  def __init__(self, data_dir=None, train_transform=None, test_transform=None, validation=False, test_size=200):
    self.data_dir = './data/covid' if data_dir is None else data_dir
    self.train_transform = train_transform
    self.test_transform = test_transform
    self.test_size = test_size
    self.validation = validation
    
    self.initialise_dataframe()
    self.initialise_datasets()
  
  def initialise_dataframe(self):
    if not (os.path.exists(f'{self.data_dir}/normal') and 
            os.path.exists(f'{self.data_dir}/covid') and 
            os.path.exists(f'{self.data_dir}/viral')):
      raise ValueError(f"Can't find the image folders at {self.data_dir}")

    normal_files = next(os.walk(f'{self.data_dir}/normal'))[2]
    normal_files = [f'/normal/{fname}' for fname in normal_files]
    viral_files  = next(os.walk(f'{self.data_dir}/viral'))[2]
    viral_files = [f'/viral/{fname}' for fname in viral_files]
    covid_files  = next(os.walk(f'{self.data_dir}/covid'))[2]
    covid_files = [f'/covid/{fname}' for fname in covid_files]
    
    if len(normal_files) < self.test_size:
      raise ValueError(f'Test size of {self.test_size} is larger than number of images in the folder {self.data_dir}/normal')
    if len(viral_files) < self.test_size:
      raise ValueError(f'Test size of {self.test_size} is larger than number of images in the folder {self.data_dir}/viral')
    if len(covid_files) < self.test_size:
      raise ValueError(f'Test size of {self.test_size} is larger than number of images in the folder {self.data_dir}/covid')
    
    test_labels = np.repeat([0, 1, 2], self.test_size)
    train_labels = [*np.repeat(0, len(normal_files) - self.test_size), 
                    *np.repeat(1, len(viral_files) - self.test_size), 
                    *np.repeat(2, len(covid_files) - self.test_size)]
    
    data_array = zip(range(len(test_labels)), 
                     [*normal_files[:self.test_size], *viral_files[:self.test_size], *covid_files[:self.test_size]],
                     test_labels
                    )
    self.test_df = pd.DataFrame(data_array, columns=['index', 'path', 'label'])
    
    data_array = zip(range(len(train_labels)), 
                     [*normal_files[self.test_size:], *viral_files[self.test_size:], *covid_files[self.test_size:]],
                     train_labels
                    )
    self.train_df = pd.DataFrame(data_array, columns=['index', 'path', 'label'])
    
    # Split into validation and train split
    if self.validation:
      validation_split = 0.1
      self.val_df = self.train_df.groupby('label').apply(pd.DataFrame.sample, n=int(validation_split * 100)).reset_index(drop=True)
      self.train_df = self.train_df.set_index('index').drop(self.val_df['index'], errors='ignore').reset_index(drop=True)
      
    self.classes = np.array(['Normal', 'Viral Pneumonia', 'Covid'])
  
  def initialise_datasets(self):
    # Set transforms for Standford Dogs data loading
    # Ensures output is 224x224 (minimum dimensions for ResNet, MobileNet etc.)
    TRAIN_TRANSFORM = T.Compose([
          T.Resize(256),
          T.CenterCrop(224),
          T.ToTensor(),
          T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
          T.RandomHorizontalFlip(),
          # T.RandomCrop(224, padding=48, padding_mode='symmetric'),
          # T.RandomErasing(p=0.3, value='random', scale=(0.02, 0.15)),
    ]) if self.train_transform is None else self.train_transform

    TEST_TRANSFORM = T.Compose([
          T.Resize(256),
          T.CenterCrop(224),
          T.ToTensor(),
          T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]) if self.test_transform is None else self.test_transform

    self.train = CovidDataset(self.train_df, self.data_dir, TRAIN_TRANSFORM, shuffle=False, train=True, classes=self.classes)
    self.query = CovidDataset(self.train_df, self.data_dir, TEST_TRANSFORM, shuffle=False, train=True, classes=self.classes)
    self.test = CovidDataset(self.test_df, self.data_dir, TEST_TRANSFORM, classes=self.classes)
    
    