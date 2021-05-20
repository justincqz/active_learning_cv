# Timing wrapper for profiling code runtime
from functools import wraps
from constants import ConfigManager

import torch
import torch.nn as nn
import time

device = ConfigManager.device

def timing(f):
  @wraps(f)
  def wrap(*args, **kw):
    start = time()
    result = f(*args, **kw)
    end = time()
    print('ook: %2.4f sec' % (end - start))
    return result
  return wrap

def log_timestamp():
  print(f'Current time: {time.asctime(time.localtime(time.time()))}')

# Added an ouput flag to print (for cleaning up during grid search)
def check_accuracy(loader, model, verbose=True):
  num_correct = 0
  num_samples = 0
  model.eval()  # set model to evaluation mode
  with torch.no_grad():
    for x, y in loader:
      x = x.to(device=device)  # move to device
      y = y.to(device=device, dtype=torch.long)
      scores = model(x)
      _, preds = scores.max(1)
      num_correct += (preds == y).sum()
      num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    if verbose:
      print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

    # Return the accuracy
    return acc * 100

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']
  
class ModelPlusOutput(nn.Module):
  def __init__(self, original_model, input_shape, output_shape, intermediate_dim=0):
    super(ModelPlusOutput, self).__init__()
    self.__name__ = original_model.__name__
    self.og_model = original_model()
    tail = []
    if intermediate_dim > 0:
      tail.append(nn.Linear(input_shape, intermediate_dim))
      input_shape = intermediate_dim
    tail.append(nn.Linear(input_shape, output_shape))
    self.out = nn.Sequential(tail)
  
  def forward(self, x):
    return self.out(self.og_model(x))

def output_wrapper_factory(original_model, input_shape, output_shape):
  def call():
    return ModelPlusOutput(original_model, input_shape, output_shape)
  
  call.__name__ = original_model.__name__
  return call

def mobilenet_wrapper_factory(mobilenet_model, output_shape):
  def call():
    net = mobilenet_model()
    num_features = net.classifier[1].in_features
    features = list(net.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, output_shape)])
    net.classifier = nn.Sequential(*features)
    return net
  
  call.__name__ = mobilenet_model.__name__
  return call

# Used to add a method to a pre-existing class
def add_method(cls):

  def decorator(func):
    @wraps(func) 
    def wrapper(self, *args, **kwargs): 
      return func(*args, **kwargs)
      
    setattr(cls, func.__name__, wrapper)
    return func
  
  return decorator