from al_sampling.uniform_random import UniformRandomSampler
from torch.utils.data import sampler, DataLoader

import numpy as np
import pandas as pd
import math

# Given boolean indicies and data, create a dataloader
def create_dataloader_from_indices(data, index, batch_size=128):
  indices = np.arange(len(data))
  sample = sampler.SubsetRandomSampler(indices[index])
  return DataLoader(data, batch_size=batch_size, sampler=sample)

# Initialises a new random seed dataset loader with the indices of the selected data
# Can have minimum samples per class requirement
def initialise_seed_dataloader(data, seed_percent=0.2, batch_size=128, min_samples=0):
  initial_indices = np.zeros(len(data), dtype=bool)
  query_amount = math.floor(seed_percent * len(data))
  idxs = []
  if min_samples > 0:
    temp_df = pd.DataFrame(zip(range(len(data.targets)), data.targets), columns=['idx', 'label'])
    idxs = np.array(temp_df.groupby('label').apply(pd.DataFrame.sample, n=min_samples).reset_index(drop=True).iloc[:,0])
    initial_indices[idxs] = True
  
  query_amount -= len(idxs)
  assert query_amount >= 0

  index = UniformRandomSampler.query_(query_amount, initial_indices, data, None, None)
  loader = create_dataloader_from_indices(data, index, batch_size)
  return loader, index

# Custom Dataloader wrapper just for TSNE visualisation
class CustomDataLoader():
  def __init__(self, image, labels):
    self.dataset = list(zip(image, labels))