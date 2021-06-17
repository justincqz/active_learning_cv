from data_processing.dataprep import DataPrep

import torchvision.transforms as T
import torchvision.datasets as dset

class FMnistDS(DataPrep):
  def __init__(self, data_dir=None, train_transform=None, test_transform=None):
    self.data_dir = './data/fmnist' if data_dir is None else data_dir
    self.train_transform = train_transform
    self.test_transform = test_transform
    self.initialise_datasets()
  
  def initialise_datasets(self):
    # Set parameters for data loading
    TRAIN_TRANSFORM = T.Compose([
          T.Grayscale(3),
          T.ToTensor(),
          T.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
          T.RandomHorizontalFlip(),
          T.RandomCrop(32, padding=4),
    ]) if self.train_transform is None else self.train_transform

    TEST_TRANSFORM = T.Compose([
          T.Grayscale(3),
          T.ToTensor(),
          T.Pad(2, padding_mode='edge'),
          T.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ]) if self.test_transform is None else self.test_transform

    # Download the Fashion MNIST dataset
    self.train = dset.FashionMNIST(self.data_dir, train=True, transform=TRAIN_TRANSFORM, download=True)
    self.test = dset.FashionMNIST(self.data_dir, train=False, transform=TEST_TRANSFORM)