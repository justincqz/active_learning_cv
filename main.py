# Flags
from training.manager import ActiveLearningComparison
from training.plotter import Plotter
from training.utils import output_wrapper_factory, mobilenet_wrapper_factory
from data_processing.dog import DogDS
from data_processing.covid import CovidDS
from al_sampling.uniform_random import UniformRandomSampler
from al_sampling.confidence import MarginSampler
from al_sampling.k_nearest import KNearestSampler
from al_sampling.mixture import MixtureSampler
from constants import ConfigManager

import argparse
import torchvision.models as models
import torch.optim as optim

if __name__ == '__main__':
  # parser = argparse.ArgumentParser(description="Active learning training environment.")
  # parser.add_argument('learning_rate', type=float, )

  ConfigManager.load_tensorboard()
  
  # Initialise the dataset
  dset = CovidDS()
  # dset.show_random_images()
  
  print("Initialised datasets.")
  
  # Setup the class which handles training and querying
  query_types = [
    {'name': 'random', 'func': UniformRandomSampler()},
    {'name': 'confidence', 'func': MarginSampler(batch_size=64)},
    {'name': 'k-nearest-confidence', 'func': KNearestSampler(batch_size=64, neighbours=15)},
    {'name': 'k-nearest-confidence-mix-15-n', 'func': MixtureSampler(KNearestSampler(batch_size=64, neighbours=15), weight=0.8, verbose=True)},
    {'name': 'k-nearest-confidence-mix-30-n', 'func': MixtureSampler(KNearestSampler(batch_size=64, neighbours=30), weight=0.8, verbose=True)},
    {'name': 'confidence-mix', 'func': MixtureSampler(MarginSampler(batch_size=64), weight=0.8, verbose=True)},
  ]

  # Setup the runner
  runner = ActiveLearningComparison(dset.train,
                                    dset.test,
                                    mobilenet_wrapper_factory(models.mobilenet_v2, 3),
                                    optim.SGD,
                                    epochs=25,
                                    learning_rate=0.02,
                                    query_percent=0.1,
                                    seed_percent=0.1,
                                    query_types=query_types,
                                    scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                                    scheduler_type='train_acc',
                                    initial_class_sample=200,
                                    batch_size=64,
                                    log_freq=1,
                                    log_level=2,
                                    run_id=300001)

  print("Initialised models.")
  # print(len(runner.cur_idx[0]))
  
  query_iterations = 3
  for i in range(query_iterations):
    print(f'Iteration: {runner.train_iter}')
    runner.run_train_and_query()
  
  runner.run_validation(iterations=10, log_freq=1, log_level=2, epochs=18)
  
  plot = Plotter()
  plt = plot.get_plots_from_file(runner.q_names, runner.save_loc + '/val-results.json')
  # plot = Plotter()
  # plt = plot.get_plots_from_multiple_files([['random', 'confidence', 'k-nearest-confidence'], 
  #                                              ['k-nearest-confidence-mix-15-n', 'k-nearest-confidence-mix-30-n', 'confidence-mix']], 
  #                                             ['./results/runs/300001'+'/val-results.json', 
  #                                              './results/runs/300003'+'/val-results.json'], eps_interval=1)
  # plt.show()