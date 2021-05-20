from al_sampling.sampler import ActiveLearningSampler
from constants import ConfigManager
import torch

import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import math

device = ConfigManager.device

class KCenterGreedy(ActiveLearningSampler):
  def __init__(self, batch_size=128, verbose=True):
    self.batch_size = batch_size
    self.verbose = verbose

  # Run through the currently known datapoints to get confidence and error data
  # TODO: Batch processing
  @staticmethod
  def get_features(data, model, batch_size=128, remove_last_layer_for_feature=True):
    # Initialise generic index list
    indices = np.arange(len(data))
    features = []

    with torch.no_grad():
      for i in indices:
        x = data[i][0].to(device=device).unsqueeze(0)
        
        if remove_last_layer_for_feature:
          feat = nn.Sequential(*list(model.children())[:-1])(x)
          feat = nn.functional.avg_pool2d(feat, feat.shape[-1]).squeeze()
          last_layer = list(model.children())[-1]
          last_layer = last_layer if hasattr(last_layer, '__iter__') else [last_layer]
          out = nn.Sequential(*last_layer)(feat).view(1, -1)
          features.append(feat.cpu())
        else:
          out = model(x)
          features.append(out.cpu())

    # Stack the list into a tensor object with shape (N, logits)
    features = torch.stack(features).squeeze()
    return features

  def query(self, query_size, known_data_idx, data, model, writer=None):
    # Initialise generic index list
    indices = np.arange(len(data))
    cluster_centers = indices[known_data_idx]

    if known_data_idx is None:
      known_data_idx = np.random.choice(indices)

    # Get the features
    features = self.get_features(data, model).unsqueeze(0).to(device) # We have to unsqueeze to make it a batch size of 1
    known_features = features[:, known_data_idx, :]

    # Calculate distance matrix
    distance_mtx = torch.cdist(features, known_features).squeeze(0)

    # Get the distance to the closest center for each feature
    min_distances = torch.min(distance_mtx, 1).values

    # Get the n data points furthest away from a cluster center
    new_indices = torch.topk(min_distances, query_size).indices.cpu().numpy()

    # Assert that none of the new indices are old
    assert len(set(new_indices).intersection(set(indices[known_data_idx]))) == 0

    # Set the flags for data known
    known_data_a = known_data_idx.copy()
    known_data_a[new_indices] = True

    # Return the new known data index
    return known_data_a
