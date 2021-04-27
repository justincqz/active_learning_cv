from al_sampling.sampler import ActiveLearningSampler
import numpy as np

class UniformRandomSampler(ActiveLearningSampler):
  @staticmethod
  def query_(query_size, known_data_idx, data, model, writer=None):
    indices = np.arange(len(data))
    choice = np.random.choice(indices[~known_data_idx], size=(query_size), replace=False)
    known_data_r = known_data_idx.copy()
    known_data_r[choice] = True
    return known_data_r

  def query(self, query_size, known_data_idx, data, model, writer=None):
    return self.query_(query_size, known_data_idx, data, model, writer)