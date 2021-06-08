from al_sampling.sampler import BaseSampler
from constants import ConfigManager
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import math

device = ConfigManager.device

class MarginSampler(BaseSampler):
  def __init__(self, weights=None, percentiles=None, margin=True, batch_size=128, verbose=True):
    self.weights = [0.6, 0.3, 0.1] if weights is None else weights
    self.percentiles = [30, 70] if percentiles is None else percentiles
    self.batch_size = batch_size
    self.margin = margin
    self.verbose = verbose

  def query(self, query_size, known_data_idx, data, model, writer=None):
    # Initialise generic index list
    indices = np.arange(len(data))
    confidence_bucket = np.zeros(len(data), dtype=float)

    # Get confidences from model
    unknown_indices = [indices[~known_data_idx][x:x+self.batch_size] for x in range(0, len(data), self.batch_size)]
    with torch.no_grad():
      for idxs in unknown_indices:
        t_stack = []
        for i in idxs:
          t_stack.append(data[i][0])
        if len(idxs) == 0:
          break
        x = torch.stack(t_stack)
        x = x.to(device=device)
        scores = model(x)
        p = F.softmax(scores, dim=1)
        if self.margin:
          top_ps = torch.topk(p, 2, dim=1).values
          top_ps = top_ps[:, 0] - top_ps[:, 1]
        else:
          top_ps = torch.max(p, dim=1).values
        confidence_bucket[idxs] = top_ps.cpu()

    # Select via buckets
    confidence_bucket = np.array(confidence_bucket)
    bounds = np.percentile(confidence_bucket[confidence_bucket != 0], self.percentiles)
    d_low = indices[(confidence_bucket <= bounds[0]) & (confidence_bucket != 0)]
    d_mid = indices[(confidence_bucket > bounds[0]) & (confidence_bucket < bounds[1])]
    d_high = indices[confidence_bucket > bounds[1]]

    low_size = math.floor(query_size * self.weights[0])
    mid_size = math.floor(query_size * self.weights[1])
    high_size = query_size - low_size -mid_size
    choice_low = np.random.choice(d_low, size=low_size, replace=False)
    choice_med = np.random.choice(d_mid, size=mid_size, replace=False)
    choice_high = np.random.choice(d_high, size=high_size, replace=False)

    # Set the flags for data known
    known_data_a = known_data_idx.copy()
    known_data_a[choice_low] = True
    known_data_a[choice_med] = True
    known_data_a[choice_high] = True

    # Fill the remainder with random
    remainder = len(choice_low) + len(choice_med) + len(choice_high)
    if remainder < query_size:
      if self.verbose:
        print(f"Couldn't fill quota of {query_size} samples with confidence brackets (got {remainder} samples). Try balancing the weights more evenly.")
      choice_random = np.random.choice(indices[~known_data_a], size=(query_size - remainder))
      known_data_a[choice_random] = True

    # Return the new known data index
    return known_data_a
