from torch import is_tensor
import numpy as np
import matplotlib.pyplot as plt

class MultiToBinaryLabel():
  def __init__(self, positive_labels=[0]):
    self.p = positive_labels

  def __call__(self, label):
    return int(label in self.p)

  def __repr__(self):
    return self.__class__.__name__ + '()'

class ToDevice():
  def __init__(self, device):
    self.device = device

  def __call__(self, label):
    assert is_tensor(label)
    return label.to(self.device)

  def __repr__(self):
    return self.__class__.__name__ + '()'
  
def show(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
  for t, m, s in zip(img, mean, std):
    t.mul_(s).add_(m)
  npimg = img.cpu().numpy()
  plt.imshow(np.transpose(npimg, (1,2,0)))
  plt.show()