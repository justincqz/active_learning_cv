from data_processing.dataprep import DataPrep
from torch.utils.data import Dataset
from scipy.io import loadmat
from PIL import Image

import torchvision.transforms as T
import torchvision.datasets as dset

import os
import torch
import pandas as pd
import numpy as np

class DogDataset(Dataset):
  """Stanford Dog Breed dataset."""

  def __init__(self, df, root_dir, transform=None, shuffle=True, classes=None, train=False):
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
    
    img_name = os.path.join(self.root_dir, self.frame.iloc[idx, -2])
    image = Image.open(img_name)
    if not image.mode == 'RGB':
      image = image.convert('RGB')
    label = int(self.frame.iloc[idx, -1])

    if self.transform:
      image = self.transform(image)

    return image, label

class DogDS(DataPrep):
  def __init__(self, data_dir=None, train_transform=None, test_transform=None, validation=False):
    self.data_dir = './data/dogs' if data_dir is None else data_dir
    self.train_transform = train_transform
    self.test_transform = test_transform
    self.validation = validation
    
    self.initialise_dataframe()
    self.initialise_datasets()
  
  def initialise_dataframe(self):
    train_annotations = loadmat(self.data_dir+'/train_list')
    size = len(train_annotations['labels'])
    data_array = zip(range(size), [s[0][0] for s in train_annotations['file_list']], [int(i[0] - 1) for i in train_annotations['labels']])
    self.train_df = pd.DataFrame(data_array, columns=['index', 'path', 'label'])
    
    # Split into validation and train split
    if self.validation:
      validation_split = 0.1
      self.val_df = self.train_df.groupby('label').apply(pd.DataFrame.sample, n=int(validation_split * 100)).reset_index(drop=True)
      self.train_df = self.train_df.set_index('index').drop(self.val_df['index'], errors='ignore').reset_index(drop=True)
      
    test_annotations = loadmat(self.data_dir+'/test_list')
    data_array = zip([s[0][0] for s in test_annotations['file_list']], [int(i[0] - 1) for i in test_annotations['labels']])
    self.test_df = pd.DataFrame(data_array, columns=['path', 'label'])
    
    self.classes = self.train_df.groupby('label').apply(pd.DataFrame.sample, n=1).reset_index(drop=True)['path']
    self.classes = self.classes.apply(lambda x: x.split('-')[1].split('/')[0])
    self.classes = np.array(self.classes)
  
  def initialise_datasets(self):
    # Set transforms for Standford Dogs data loading
    # Ensures output is 224x224 (minimum dimensions for ResNet, MobileNet etc.)
    TRAIN_TRANSFORM = T.Compose([
          T.Resize(256),
          T.CenterCrop(224),
          T.ToTensor(),
          T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
          T.RandomHorizontalFlip(),
          T.RandomCrop(224, padding=48, padding_mode='symmetric'),
          T.RandomErasing(p=0.3, value='random', scale=(0.02, 0.15)),
    ]) if self.train_transform is None else self.train_transform

    TEST_TRANSFORM = T.Compose([
          T.Resize(256),
          T.CenterCrop(224),
          T.ToTensor(),
          T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]) if self.test_transform is None else self.test_transform

    self.train = DogDataset(self.train_df, f'{self.data_dir}/images', TRAIN_TRANSFORM, shuffle=False, train=True, classes=self.classes)
    self.test = DogDataset(self.test_df, f'{self.data_dir}/images', TEST_TRANSFORM, classes=self.classes)
    
    