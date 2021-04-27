import abc

class ActiveLearningSampler(object):
  __metaclass__ = abc.ABCMeta
  
  @abc.abstractmethod
  def query(self, query_size, known_data_idx, data, model, writer=None):
    return
