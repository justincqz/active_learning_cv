from ast import literal_eval
import os
import numpy as np
import matplotlib.pyplot as plt

class Plotter():
  def __init__(self):
    self.colours = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
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
    plt_test_acc  = self.get_plot(types, f_test, subplot_idx=3, eps_interval=eps_interval, subject='Test Accuracy', y_low=60, y_high=95)

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