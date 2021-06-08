from data_processing.covid import CovidDS
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt

def show(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
  for t, m, s in zip(img, mean, std):
    t.mul_(s).add_(m)
  npimg = img.cpu().numpy()
  plt.imshow(np.transpose(npimg, (1,2,0)))

ds = CovidDS()

imgs = []
cls  = []
for i in range(6200, 6216):
  imgs.append(ds.train[i][0])
  cls.append(ds.train[i][1])

imgs = make_grid(imgs)

plt.figure(figsize=(16, 4))
show(imgs)

print([ds.test.classes[c] for c in cls])
plt.show()

