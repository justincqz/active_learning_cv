from al_sampling.sampler import ActiveLearningSampler
from constants import ConfigManager

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import math

import contextlib

@contextlib.contextmanager
def null_ctx():
  yield None
    
device = ConfigManager.device

def enum(*args, **named):
  enums = dict(zip(args, [object() for _ in range(len(args))]), **named)
  return type('Enum', (), enums)

class SamplerFactory(ActiveLearningSampler):
  entropy = enum('confidence', 'margin', 'gradients')
  diversity = enum('coreset', 'kmpp', 'random')

  def __init__(self, options=None) -> None:
    super().__init__()
    self.scoring_method = options.get('entropy', default=self.entropy.confidence)
    self.diversity_method = options.get('diversity', default=self.diversity.random)
    self.batch_size = options.get('batch_size', default=128)
    self.diversity_weighted = options.get('diversity_weighted', default=False) # If the diversity metric should be weighted by the scoring function
    self.diversity_mix = options.get('diversity_mix', default=0.2) # Mix between top-n and a diversity metric
    self.options = options

  def get_metrics(self, data, model):
    indices = np.arange(len(data))
    features = []
    scores = []
    
    feature_layer = nn.Sequential(*list(model.children())[:-1])
    last_module = list(model.children())[-1]
    last_module = last_module if hasattr(last_module, '__iter__') else [last_module]
    output_layer = nn.Sequential(*last_module)
    
    with torch.no_grad() if self.scoring_method is not self.entropy.gradients else null_ctx():
      for idxs in range([indices[x:x+self.batch_size] for x in range(0, len(data), self.batch_size)]):
        t_stack = []
        for i in idxs:
          t_stack.append(data[i][0])
        if len(idxs) == 0:
          break
        x = torch.stack(t_stack)
        x = x.to(device=device)
        
        feature = feature_layer(x)
        features.append(feature.squeeze().cpu())
        logits = output_layer(feature)
        
        if self.scoring_method is self.entropy.confidence:
          probabilities = F.softmax(logits, dim=1)
          top_ps = torch.max(probabilities, dim=1).values
          scores.append(top_ps.cpu())
          
        elif self.scoring_method is self.entropy.margin:
          probabilities = F.softmax(logits, dim=1)
          top_ps = torch.topk(probabilities, 2, dim=1).values
          top_ps = top_ps[:, 0] - top_ps[:, 1]
          scores.append(top_ps.cpu())
          
        elif self.scoring_method is self.entropy.gradients:
          probabilities = F.softmax(logits, dim=1)
          fake_target = torch.max(probabilities, dim=1).indices
          errors = F.cross_entropy(logits, fake_target, reduction='none')
          scores.append(errors.cpu().detach())
          
    features = torch.cat(features, dim=0).squeeze()
    scores = torch.cat(scores, dim=0).squeeze()
    
    return scores, features
  
  def coreset(self, query_size, known_data_idx, scores, features, distance_metric='euclidean', weighted_by_score=False, **kwargs):
    # Initialise generic index list
    indices = np.arange(len(features))
    
    # Unsqueeze to simulate batch size of 1
    features = features.unsqueeze(0).to(device)

    # Get the features
    known_features = features[:, known_data_idx, :]

    # Calculate distance matrix
    if distance_metric is 'euclidean':
      distance_mtx = torch.cdist(features, known_features).squeeze(0)
    elif distance_metric is 'cosine':
      distance_mtx = None
    else:
      raise ValueError("Distance metric not supported, choose either 'euclidean' or 'cosine'.")

    # Get the distance to the closest center for each feature
    min_distances = torch.min(distance_mtx, 1).values

    # Weigh the distances by scores
    if weighted_by_score:
      min_distances = min_distances * scores
    
    # Get the n data points furthest away from a cluster center
    new_indices = torch.topk(min_distances, query_size).indices.cpu().numpy()

    # Assert that none of the new indices are old
    assert len(set(new_indices).intersection(set(indices[known_data_idx]))) == 0

    # Set the flags for data known
    known_data_a = known_data_idx.copy()
    known_data_a[new_indices] = True

    # Return the new known data index
    return known_data_a
  
  def top_n(self, query_size, known_data_idx, scores, descending=False):
    # Initialise generic index list
    indices = np.arange(len(known_data_idx))

    # Get the top performers
    sorted_indices = torch.argsort(scores, descending=descending)
    
    # Add the top n which do not occur in the already seen indices
    new_indices = []
    known_indices = set(indices[known_data_idx])
    for s in sorted_indices:
      if len(new_indices) == query_size:
        break
      if s not in known_indices:
        new_indices.append(s)

    assert len(new_indices) == query_size
    
    # Set the flags for data known
    known_data_a = known_data_idx.copy()
    known_data_a[new_indices] = True

    return known_data_a
  
  def kmpp(self):
    pass
  
  def random(self, query_size, known_data_idx):
    indices = np.arange(len(known_data_idx))
    choice = np.random.choice(indices[~known_data_idx], size=(query_size), replace=False)
    known_data_r = known_data_idx.copy()
    known_data_r[choice] = True
    return known_data_r

  # TODO: Verbose
  def query(self, query_size, known_data_idx, data, model, writer=None):
    scores, features = self.get_metrics(data, model)
    if self.scoring_method is self.diversity.coreset:
      return self.coreset(query_size, known_data_idx, scores, features, **self.options)
    elif self.scoring_method is self.diversity.random:
      active_n = int(query_size * (1 - self.diversity_mix))
      passive_n = query_size - active_n
      if active_n != 0:
        known_data_r = self.top_n(active_n, known_data_idx, scores, descending=self.scoring_method is self.entropy.gradients)
      if passive_n != 0:
        known_data_r = self.random(passive_n, known_data_r)
      return known_data_r
    elif self.scoring_method is self.diversity.kmpp:
      raise SystemError('Unimplemented.')
    else:
      raise ValueError("Chosen diversity metric not supported.")
