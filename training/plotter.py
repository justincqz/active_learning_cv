import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from ast import literal_eval
from sklearn.manifold import TSNE
from constants import ConfigManager
from training.utils import log_timestamp

device = ConfigManager.device

class Plotter():
  def __init__(self):
    self.colours = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'black', 'gray']
    self.figsize = (18, 5)
    self.plot = None

  def get_plot(self, types, res, subplot_idx, eps_interval=1, subject='Train Accuracy', y_low=None, y_high=None):
    epochs = np.array(range(0, len(res[0][0]) * eps_interval, eps_interval))

    if self.plot is None:
      self.plot = plt.figure(figsize=self.figsize)
    else:
      plt.subplot(1, 3, subplot_idx)
    plt.title(f'Average {subject}')

    for i in range(len(types)):
      avg_res = np.average(res[i], axis=0)

      for iters in range(len(res[0])):
        plt.plot(epochs, res[i][iters], color=self.colours[i], alpha=0.06)

      plt.plot(epochs, avg_res, color=self.colours[i], alpha=0.8, label=f'Average {subject} for {types[i]}')

    plt.xlabel('Epochs')
    plt.ylabel(subject)
    plt.xlim(0, epochs[-1])
    if not (y_low is None and y_high is None):
      plt.ylim(y_low, y_high)
    plt.legend()

    return plt

  def get_loss_and_accuracy_plots(self, types, results, eps_interval=1):
    self.plot = plt.figure(figsize=self.figsize)

    f_loss  = np.array([[[x['loss']      for x in xs] for xs in res_t] for res_t in results])
    f_train = np.array([[[x['train_acc'] for x in xs] for xs in res_t] for res_t in results])
    f_test  = np.array([[[x['test_acc']  for x in xs] for xs in res_t] for res_t in results])

    plt_loss      = self.get_plot(types, f_loss, subplot_idx=1, eps_interval=eps_interval, subject='Loss')
    plt_train_acc = self.get_plot(types, f_train, subplot_idx=2, eps_interval=eps_interval, subject='Train Accuracy')
    plt_test_acc  = self.get_plot(types, f_test, subplot_idx=3, eps_interval=eps_interval, subject='Test Accuracy', y_low=62.5, y_high=92.5)

    plt.tight_layout(1)
    return plt

  def get_plots_from_file(self, types, file_loc, eps_interval=1):
    if not os.path.isfile(file_loc):
      return
    f = open(file_loc,'r')
    res_data = ''.join(f.readlines())
    f.close()
    res_data = literal_eval(res_data)
    results = list(res_data['results'].values())
    
    return self.get_loss_and_accuracy_plots(types, results, eps_interval)

  def get_plots_from_multiple_files_joined(self, types, files, eps_interval=1):
    joined_results = {i: [] for i in range(len(types))}
    for i in range(len(files)):
      if not os.path.isfile(files[i]):
        raise FileExistsError(f'{files[i]} not found.')
      f = open(files[i],'r')
      res_data = ''.join(f.readlines())
      f.close()
      res_data = literal_eval(res_data)

      assert len(res_data['results']) == len(types)
      for j in range(len(res_data['results'])):
        joined_results[j].extend(res_data['results'][j])
    
    return self.get_loss_and_accuracy_plots(types, list(joined_results.values()), eps_interval)

  def get_plots_from_multiple_files(self, types, files, eps_interval=1):
    assert len(types) == len(files)

    results = []
    for fi in range(len(files)):
      if not os.path.isfile(files[fi]):
        raise FileExistsError(f'{files[fi]} not found.')
      f = open(files[fi],'r')
      res_data = ''.join(f.readlines())
      f.close()
      res_data = literal_eval(res_data)
      results.append(list(res_data['results'].values()))

    return self.get_loss_and_accuracy_plots([j for i in types for j in i], [j for i in results for j in i], eps_interval)

  def show_plot(self):
    self.plot.show()


# Plots the TSNE using the model
class TSNEPlotter():
  def __init__(self, query_names, cur_idx_path, prev_idx_path, model_paths, model, data, save=None):
    if not os.path.isfile(prev_idx_path):
      raise ValueError(f'Index file not found at {prev_idx_path}.')
    if not os.path.isfile(cur_idx_path):
      raise ValueError(f'Index file not found at {cur_idx_path}.')

    self.models = []
    for model_path in model_paths:
      if not os.path.isfile(model_path):
        raise ValueError(f'Model file not found at {model_path}.')
      try:
        m = model()
        m.load_state_dict(torch.load(model_path))
        m.to(device)
        self.models.append(m)
      except Exception:
        raise ValueError(f'Unable to load model from path {model_path}.')

    prev_idx_zip = np.load(prev_idx_path)
    cur_idx_zip  = np.load(cur_idx_path)

    self.prev_idx = [prev_idx_zip[str(i)] for i in range(len(query_names))]
    self.cur_idx  = [cur_idx_zip[str(i)] for i in range(len(query_names))]
    self.query_names = query_names
    self.data = data
    self.colours = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'black', 'brown', 'lime', 'cyan']
    self.figures = []

    # If a save directory is given, save tsne plots generated to this location
    if save is not None:
      os.makedirs(save, exist_ok=True)

    self.save = save

  def get_features(self, model, remove_last_layer_for_feature=False, batch_size=256):
    # Initialise generic index list
    indices = np.arange(len(self.data))
    features = []
    classes = []

    with torch.no_grad():
      for i in [indices[x:x+batch_size] for x in range(0, len(self.data), batch_size)]:
        x = []
        for i_ in i:
          x.append(self.data[i_][0].to(device=device))
          y = self.data[i_][1]
          classes.append(y)
        x = torch.stack(x).to(device)
        
        if remove_last_layer_for_feature:
          feat = nn.Sequential(*list(model.children())[:-1])(x)
          feat = nn.functional.avg_pool2d(feat, feat.shape[-1]).squeeze()
          last_layer = list(model.children())[-1]
          last_layer = last_layer if hasattr(last_layer, '__iter__') else [last_layer]
          # out = nn.Sequential(*last_layer)(feat).view(1, -1)
          features.extend(feat.cpu())
        else:
          out = model(x)
          features.extend(out.cpu())

    # Stack the list into a tensor object with shape (N, logits)
    features = torch.stack(features).squeeze().numpy()
    classes = torch.LongTensor(classes).numpy()
    return features, classes

  def generate_tsne(self, idx=0, verbose=True, alpha=[0.2, 0.3, 0.9]):
    if isinstance(idx, int):
      idx = [idx]
    elif not hasattr(idx, "__iter__") and isinstance(next(idx.__iter__()), int):
      raise ValueError(f'Expect indices argument to be either a single integer or a list/iterable of ints, instead got {type(idx)}')
    
    for index in idx:
      if verbose:
        print(f'Generating TSNE plot for {self.query_names[index]}')
        print('Getting features from data.')
        log_timestamp()
      xs, ys = self.get_features(self.models[index])
      
      if verbose:
        print('Generating TSNE representation.')
        log_timestamp()
      transformed_xs = TSNE(n_components=2).fit_transform(xs)

      if verbose:
        print('Generating scatter plot.')
        log_timestamp()

      # We make two subplots, one for comparing known vs unknown and one for comparing classification labels
      # First we look at the classification labels
      fig = plt.figure()
      ax = fig.add_subplot(1, 2, 1)
      plt.title(f'TSNE plots for {self.query_names[index]} Query wrt Labels')
      groups = [[] for _ in range(len(set(ys)))]
      
      # Add the data samples to the correct labelled groups
      for i in range(len(ys)):
        groups[ys[i]].append(transformed_xs[i])

      for i in range(len(groups)):
        d = np.array(groups[i])
        ax.scatter(d[:,0], d[:,1], alpha=alpha[1], c=self.colours[i], edgecolors='none', s=30, label=i)
      
      # Next we evaluate which points the active learing algorithm chose
      ax = fig.add_subplot(1, 2, 2)
      plt.title(f'TSNE plots for {self.query_names[index]} Query wrt Known/Unknown/New')
      indices = np.arange(len(ys))
      unknown = transformed_xs[indices[~self.cur_idx[index]]]
      known   = transformed_xs[indices[self.prev_idx[index]]]
      new     = transformed_xs[indices[self.prev_idx[index] ^ self.cur_idx[index]]]

      groups = [unknown, known, new]
      group_labels = ['unknown', 'known', 'new']

      for i in range(len(groups)):
        d = np.array(groups[i])
        ax.scatter(d[:,0], d[:,1], alpha=alpha[i], c=self.colours[i], edgecolors='none', s=30, label=group_labels[i], zorder=i)
      
      plt.legend(loc=2)
      self.figures.append(fig)
      if self.save is None:
        plt.show()
      else:
        plt.savefig(f'{self.save}/{self.query_names[index]}.png')

def get_tsne_features(data, model, remove_last_layer_for_feature=False, batch_size=256):
  # Initialise generic index list
  indices = np.arange(len(data))
  features = []
  classes = []

  with torch.no_grad():
    for i in [indices[x:x+batch_size] for x in range(0, len(data), batch_size)]:
      x = []
      for i_ in i:
        x.append(data[i_][0].to(device=device))
        y = data[i_][1]
        classes.append(y)
      x = torch.stack(x).to(device)
      
      if remove_last_layer_for_feature:
        feat = nn.Sequential(*list(model.children())[:-1])(x)
        feat = nn.functional.avg_pool2d(feat, feat.shape[-1]).squeeze()
        last_layer = list(model.children())[-1]
        last_layer = last_layer if hasattr(last_layer, '__iter__') else [last_layer]
        # out = nn.Sequential(*last_layer)(feat).view(1, -1)
        features.extend(feat.cpu())
      else:
        out = model(x)
        features.extend(out.cpu())

  # Stack the list into a tensor object with shape (N, logits)
  features = torch.stack(features).squeeze().numpy()
  classes = torch.LongTensor(classes).numpy()

  # Generate the TSNE plot
  transformed_xs = TSNE(n_components=2).fit_transform(features)
  
  return features, transformed_xs, classes 