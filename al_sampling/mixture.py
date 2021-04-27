from al_sampling.uniform_random import UniformRandomSampler
from al_sampling.sampler import ActiveLearningSampler

class MixtureSampler(ActiveLearningSampler):
  def __init__(self, first_sampler, second_sampler=None, weight=0.5, verbose=True):
    self.first_sampler = first_sampler
    self.second_sampler = UniformRandomSampler() if second_sampler is None else second_sampler
    self.weight = weight
    self.verbose = verbose
    
  def query(self, query_size, known_data_idx, data, model, writer=None):
    first_size = int(query_size * self.weight)
    second_size = query_size - first_size
    known_data_r = self.first_sampler.query(first_size, known_data_idx, data, model, writer)
    known_data_r = self.second_sampler.query(second_size, known_data_r, data, model, writer)
    return known_data_r