from data_processing.utils import show
from torchvision.utils import make_grid

import abc
import random
import matplotlib.pyplot as plt

class DataPrep(object):
  __metaclass__ = abc.ABCMeta
  
  @abc.abstractmethod
  def initialise_datasets(self):
    return
  
  def get_datasets(self):
    assert self.train is not None
    assert self.test is not None
    return self.train, self.test
  
  def show_random_images(self, number=16):
    size = len(self.train)
    indices = [random.randint(0, size) for _ in range(number)]
    
    imgs = []
    classes  = []
    for i in indices:
      imgs.append(self.train[i][0])
      classes.append(self.train[i][1])

    imgs = make_grid(imgs)

    plt.figure(figsize=(16, 4))
    print(self.train.classes[classes])
    show(imgs)
