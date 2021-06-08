import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from al_sampling.sampler import BaseSampler
from constants import ConfigManager

device = ConfigManager.device

class KNearestSampler(BaseSampler):
  def __init__(self, criteria='conf', batch_size=128, neighbours=5, verbose=True):
    if not (criteria == 'conf' or criteria == 'loss'):
      raise ValueError("Invalid criteria selected, choose either 'conf' or 'loss'.")
    self.criteria = criteria
    self.batch_size = batch_size
    self.neighbours = neighbours
    self.verbose = verbose
    
  # Run through the currently known datapoints to get confidence and error data
  # TODO: Batch processing
  @staticmethod
  def get_statistics(data, model, known_data_idx, batch_size=128, remove_last_layer_for_feature=True):
    # Initialise generic index list
    indices = np.arange(len(data))
    known_indices = set(indices[known_data_idx])
    out_stats = {}
    features = []

    with torch.no_grad():
      for i in indices:
        x = data[i][0].to(device=device).unsqueeze(0)
        y = torch.LongTensor([data[i][1]]).to(device)
        
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
          
        if i in known_indices:
          loss = F.cross_entropy(out, y)
          p = F.softmax(out, dim=1)
          conf = torch.topk(p, 2, dim=1).values
          conf = conf[:, 0] - conf[:, 1]

          out_stats[i] = {'loss': loss.tolist(), 'confidence': conf.tolist()[0]}

    # Stack the list into a tensor object with shape (N, logits)
    features = torch.stack(features).squeeze()
    return out_stats, features

  @staticmethod
  def cosine_distance_k_nearest(ref, query, reference_indices, query_size, batch_size=512, neighbours=3):
    '''
    Calculate the distance matrix between all tensors using cosine similarity.

    Input: 
    - ref <Tensors> (N1, Dim) - For similarity check
    - query <Tensors> (N2, Dim) - For similarity check
    - reference_indices <List<int>> - The indicies of the references
    - query_size <int> - How many indices to return
    - batch_size <int> (Optional) - Process in batches for memory optimisation
    '''
    pad_query = query.shape[0] % batch_size
    query = torch.cat([query, torch.zeros(pad_query, query.shape[1])])
    selected = set()

    for i in range(ref.shape[0]):
      temp = torch.zeros(query.shape[0])
      
      for j in range(int(query.shape[0] / batch_size)):
        out = F.cosine_similarity(ref[i].unsqueeze(0).to(device), query[(j * batch_size):((j + 1) * batch_size)].to(device), dim=1)
        temp[(j * batch_size):((j + 1) * batch_size)] = out.cpu()
      selected.update(reference_indices[temp.topk(neighbours, sorted=False)[1].tolist()])
      if len(selected) > query_size:
        break
    
    return list(selected)[:query_size]

  def query(self, query_size, known_data_idx, data, model, writer=None):
    # Initialise generic index list
    indices = np.arange(len(data))

    if self.verbose:
      print("Generating statistics for known datapoints.")
    known_stats, xs = self.get_statistics(data, model, known_data_idx, batch_size=self.batch_size)

    # Sort the datapoints to be used for k-nearest neighbours based on criteria type
    if self.criteria == 'conf':
      confidences = np.ones(len(data))
      confidences[list(known_stats.keys())] = [i['confidence'] for i in known_stats.values()]
      seed_args = np.argsort(confidences)
    elif self.criteria == 'loss':
      losses = np.zeros(len(data))
      losses[list(known_stats.keys())] = [i['loss'] for i in known_stats.values()]
      seed_args = np.argsort(-losses)
    else:
      raise ValueError("Invalid criteria selected, choose either 'conf' or 'loss'.")

    if self.verbose:
      print("Calculating k-nearest for datapoints.")

    # Add datapoints until we reached
    unknown_indices = indices[~known_data_idx]
    chosen_indices = self.cosine_distance_k_nearest(xs[seed_args], xs[unknown_indices], unknown_indices, query_size, neighbours=self.neighbours)
    
    # Set the flags for data known
    known_data_a = known_data_idx.copy()
    known_data_a[chosen_indices] = True

    # Fill randoms if we miss the quota
    if len(chosen_indices) < query_size:
      choice = np.random.choice(indices[~known_data_a], size=(query_size - len(chosen_indices)), replace=False)
      known_data_a[choice] = True

      if self.verbose:
        print(f"Couldn't fill quota of {query_size} samples with {self.neighbours} neighbours ({len(chosen_indices)} samples). Try increasing neighbour count.")

    # Save a figure of which nodes are picked
    # if not writer is None:
    #   ys = np.zeros(len(ys), dtype=int)
      # ys[chosen_indices] = 1
      # ys[known_data_idx] = 2
      # fig = plot_tsne(tsnes, ys)
      # writer.add_figure('Plot/Query_Emb', fig)
    return known_data_a