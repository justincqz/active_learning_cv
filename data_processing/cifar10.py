from data_processing.dataprep import DataPrep

import torchvision.transforms as T
import torchvision.datasets as dset

class Cifar10DS(DataPrep):
  def __init__(self, data_dir=None, train_transform=None, test_transform=None):
    self.data_dir = './data/cifar10' if data_dir is None else data_dir
    self.train_transform = train_transform
    self.test_transform = test_transform
  
  def initialise_datasets(self):
    # Set parameters for data loading
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

    TRAIN_TRANSFORM = T.Compose([
          T.ToTensor(),
          T.Normalize(CIFAR10_MEAN,CIFAR10_STD),
          T.RandomHorizontalFlip(),
          T.RandomCrop(32, padding=4),
    ]) if self.train_transform is None else self.train_transform

    TEST_TRANSFORM = T.Compose([
          T.ToTensor(),
          T.Normalize(CIFAR10_MEAN,CIFAR10_STD),
    ]) if self.test_transform is None else self.test_transform

    # Download the Cifar10 dataset
    self.train = dset.CIFAR10(self.data_dir, train=True, transform=TRAIN_TRANSFORM, download=True)
    self.test = dset.CIFAR10(self.data_dir, train=False, transform=TEST_TRANSFORM)