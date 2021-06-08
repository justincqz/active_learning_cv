from al_sampling.sampler import BaseSampler
from al_sampling.utils import PCA
from constants import ConfigManager
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import contextlib

@contextlib.contextmanager
def null_ctx():
  yield None

device = ConfigManager.device

def enum(*args, **named):
  enums = dict(zip(args, [object() for _ in range(len(args))]), **named)
  return type('Enum', (), enums)

class ActiveLearningSampler(BaseSampler):
  entropy = enum('confidence', 'margin', 'gradients', 'custom')
  diversity = enum('coreset', 'knearest', 'random', 'custom')

  def __init__(self, options=None, verbose=True) -> None:
    super().__init__()
    self.scoring_method = options.get('entropy', self.entropy.confidence)
    self.diversity_method = options.get('diversity', self.diversity.random)
    self.batch_size = options.get('batch_size', 128)
    self.diversity_mix = options.get('diversity_mix', 0.2) # Mix between top-n and a diversity metric
    self.use_pca = options.get('use_pca', False)
    self.pl_amount = options.get('passive_learning', 0)
    self.custom_scoring = options.get('scoring_function', None)
    self.custom_diversity = options.get('diversity_function', None)
    self.options = options
    self.verbose = verbose

  def get_metrics(self, data, model):
    if self.scoring_method is self.entropy.custom:
      return self.wrapper_scoring_function(data, model, self.custom_scoring)

    indices = np.arange(len(data))
    features = []
    scores = []

    feature_layers = [*list(model.children())[:-1]]
    last_module = list(model.children())[-1]
    if hasattr(last_module, '__iter__'):
      def feature_layer(x):
        feat = nn.Sequential(*feature_layers)(x).squeeze()
        feat = nn.functional.avg_pool2d(feat, feat.shape[-1]).squeeze()
        feat = nn.Sequential(*list(last_module[:-1]))(feat)
        return feat
      output_layer = last_module[-1]
    else:
      feature_layer = nn.Sequential(*feature_layers)
      output_layer = nn.Sequential(last_module)

    with torch.no_grad() if self.scoring_method is not self.entropy.gradients else null_ctx():
      for idxs in [indices[x:x+self.batch_size] for x in range(0, len(data), self.batch_size)]:
        t_stack = []
        for i in idxs:
          t_stack.append(data[i][0])
        if len(idxs) == 0:
          break
        x = torch.stack(t_stack).to(device=device).detach()
        if self.scoring_method is self.entropy.gradients:
          x.requires_grad = True
          x.retain_grad()
        
        feature = feature_layer(x).squeeze()
        features.append(feature.detach().cpu())
        logits = output_layer(feature)
        probabilities = F.softmax(logits, dim=1)
        
        if self.scoring_method is self.entropy.confidence:
          num_labels = probabilities.shape[-1]
          top_ps = (1 - (torch.max(probabilities, dim=1).values)) * (num_labels / (num_labels - 1))
          scores.append(top_ps.cpu())
          
        elif self.scoring_method is self.entropy.margin:
          top_ps = torch.topk(probabilities, 2, dim=1).values
          top_ps = 1 - (top_ps[:, 0] - top_ps[:, 1])
          scores.append(top_ps.cpu())

        elif self.scoring_method is self.entropy.gradients:
          fake_target = torch.max(probabilities, dim=1).indices
          errors = F.cross_entropy(logits, fake_target)
          errors.backward()
          grad = torch.sum(torch.abs(x.grad), tuple(range(1, len(x.shape))))
          scores.append(grad.cpu().detach())

    features = torch.cat(features, dim=0).squeeze()
    scores = torch.cat(scores, dim=0).squeeze()
    
    return scores, features
  
  def coreset(self, query_size, known_data_idx, scores, features, distance_metric='euclidean', weighted_by_score=True, **kwargs):
    # Initialise generic index list
    indices = np.arange(len(features))
    
    # Unsqueeze to simulate batch size of 1
    features = features.unsqueeze(0)

    # Get the features
    known_features = features[:, known_data_idx, :]

    # Calculate distance matrix
    if distance_metric is 'euclidean':
      distance_mtx = torch.cdist(features, known_features).squeeze(0)
    elif distance_metric is 'manhattan':
      distance_mtx = torch.cdist(features, known_features, p=1).squeeze(0)
    elif distance_metric is 'cosine':
      known_features = known_features.squeeze(0)
      features = features.squeeze(0)
      distance_mtx = cosine_similarity(known_features, features)
      distance_mtx = torch.tensor(distance_mtx, device=device)
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

    # Get the top uncertain entries
    sorted_indices = np.argsort(scores.cpu().numpy())
    
    # Add the top n which do not occur in the already seen indices
    new_indices = set()
    known_indices = set(indices[known_data_idx])
    for s in sorted_indices:
      if len(new_indices) == query_size:
        break
      if s not in known_indices:
        new_indices.add(s)

    new_indices = list(new_indices)
    assert len(new_indices) == query_size
    
    # Set the flags for data known
    known_data_a = known_data_idx.copy()
    known_data_a[new_indices] = True

    return known_data_a

  def knearest(self, query_size, known_data_idx, scores, features, distance_metric='euclidean', num_neighbours=15, **kwargs):
    # Initialise generic index list
    indices = np.arange(len(features))
    
    # Unsqueeze to simulate batch size of 1
    features = features.unsqueeze(0)

    # Get the features
    known_features = features[:, known_data_idx, :]

    # Calculate distance matrix
    if distance_metric is 'euclidean':
      distance_mtx = torch.cdist(known_features, features).squeeze(0).cpu().numpy()
    elif distance_metric is 'manhattan':
      distance_mtx = torch.cdist(known_features, features, p=1).squeeze(0).cpu().numpy()
    elif distance_metric is 'cosine':
      known_features = known_features.squeeze(0)
      features = features.squeeze(0)
      distance_mtx = cosine_similarity(known_features, features)
    else:
      raise ValueError("Distance metric not supported, choose either 'euclidean', 'manhattan' or 'cosine'.")

    max_neighbours = min(num_neighbours, len(distance_mtx[0]))
    known_indices = set(indices[known_data_idx])
    sorted_scores = np.argsort(scores.cpu().numpy())
    seed_points = np.argsort([idx for idx in sorted_scores if idx in known_indices])

    if self.verbose:
      print("Calculating k-nearest for datapoints.")

    chosen_indices = set()
    # Add datapoints until we reached
    for i in seed_points:
      if len(chosen_indices) >= query_size:
        break
      top_indices = np.argsort(distance_mtx[i])
      new_points = [j for j in top_indices if (j not in known_indices) and (j not in chosen_indices)]
      chosen_indices.update(set(new_points[:max_neighbours]))

    # Convert set back to list for indexing
    chosen_indices = list(chosen_indices)[:query_size]
    
    # Set the flags for data known
    known_data_a = known_data_idx.copy()
    known_data_a[chosen_indices] = True

    # Fill randoms if we miss the quota
    if len(chosen_indices) < query_size:
      choice = np.random.choice(indices[~known_data_a], size=(query_size - len(chosen_indices)), replace=False)
      known_data_a[choice] = True

      if self.verbose:
        print(f"Couldn't fill quota of {query_size} samples with {max_neighbours} neighbours ({len(chosen_indices)} samples). Try increasing neighbour count.")

    return known_data_a
  
  def random(self, query_size, known_data_idx, mix_amount=1.0):
    rand_amount = int(query_size * mix_amount)
    indices = np.arange(len(known_data_idx))
    choice = np.random.choice(indices[~known_data_idx], size=(rand_amount), replace=False)
    known_data_r = known_data_idx.copy()
    known_data_r[choice] = True
    return known_data_r, rand_amount

  def wrapper_scoring_function(self, data, model, function):
    return function(data, model, **self.options)

  def wrapper_diversity_function(self, query_size, known_data_idx, scores, features, function):
    return function(query_size, known_data_idx, scores, features, **self.options)

  def query(self, query_size, known_data_idx, data, model, writer=None):
    # Scoring
    if self.verbose:
      print('Getting metrics (score and features).')
    scores, features = self.get_metrics(data, model)
    if self.use_pca:
      if self.verbose:
        print('Performing PCA.')
      features = PCA().fit_transform(features)

    # Passive learning mixture
    if self.pl_amount > 0:
      known_data_idx_, pl_count = self.random(query_size, known_data_idx, mix_amount=self.pl_amount)
      if self.verbose:
        print(f'Apply passive learning for {pl_count} samples.')
      query_size = query_size - pl_count
    else:
      known_data_idx_ = known_data_idx

    # Sampling
    if self.verbose:
      print('Apply diversity sampling.')
    if self.diversity_method is self.diversity.coreset:
      return self.coreset(query_size, known_data_idx_, scores, features, **self.options)
    if self.diversity_method is self.diversity.knearest:
      return self.knearest(query_size, known_data_idx_, scores, features, **self.options)
    elif self.diversity_method is self.diversity.random:
      active_n = int(query_size * (1 - self.diversity_mix))
      passive_n = query_size - active_n
      if active_n != 0:
        if self.verbose:
          print(f'Sampling {active_n} samples from top-n.')
        known_data_idx_ = self.top_n(active_n, known_data_idx_, scores, descending=self.scoring_method is self.entropy.gradients)
      if passive_n != 0:
        if self.verbose:
          print(f'Sampling {passive_n} samples from random.')
        known_data_idx_, _ = self.random(passive_n, known_data_idx_)
      return known_data_idx_
    elif self.diversity_method is self.diversity.custom:
      return self.wrapper_diversity_function(query_size, known_data_idx, scores, features, self.custom_diversity)
    else:
      raise ValueError("Chosen diversity metric not supported.")
