# Flags
from training.manager import ActiveLearningComparison
from training.plotter import Plotter, TSNEPlotter
from training.utils import output_wrapper_factory, mobilenet_wrapper_factory
from data_processing.cifar10 import Cifar10DS
from data_processing.dog import DogDS
from data_processing.covid import CovidDS
from al_sampling.sample_factory import SamplerFactory
from al_sampling.uniform_random import UniformRandomSampler
from al_sampling.confidence import MarginSampler
from al_sampling.k_nearest import KNearestSampler
from al_sampling.mixture import MixtureSampler
from al_sampling.k_center_greedy import KCenterGreedy
from constants import ConfigManager

import torchvision.models as models
from models import ResNet20
import torch.optim as optim

if __name__ == '__main__':
  # parser = argparse.ArgumentParser(description="Active learning training environment.")
  # parser.add_argument('learning_rate', type=float, )

  ConfigManager.load_tensorboard()
  
  # Initialise the dataset
  # dset = CovidDS()
  dset = Cifar10DS()
  # dset.show_random_images()
  
  print("Initialised datasets.")
  
  # Setup the class which handles training and querying
  # query_types = [
  #   {'name': 'random', 'func': UniformRandomSampler()},
  #   {'name': 'confidence', 'func': MarginSampler(batch_size=256)},
  #   {'name': 'k-nearest-confidence', 'func': KNearestSampler(batch_size=256, neighbours=15)},
  #   {'name': 'k-center', 'func': KCenterGreedy(batch_size=256)},
  #   {'name': 'k-nearest-confidence-mix-15-n', 'func': MixtureSampler(KNearestSampler(batch_size=256, neighbours=15), weight=0.8, verbose=True)},
  #   {'name': 'k-nearest-confidence-mix-30-n', 'func': MixtureSampler(KNearestSampler(batch_size=256, neighbours=30), weight=0.8, verbose=True)},
  #   {'name': 'confidence-mix', 'func': MixtureSampler(MarginSampler(batch_size=256), weight=0.8, verbose=True)},
  #   {'name': 'k-center-mix', 'func': MixtureSampler(KCenterGreedy(batch_size=256), weight=0.8, verbose=True)},
  # ]

  query_types = [
    {'name': 'conf-weighted-coreset-euclidean-pca', 'func': SamplerFactory({
      'entropy': SamplerFactory.entropy.margin,
      'diversity': SamplerFactory.diversity.coreset,
      'batch_size': 128,
      'use_pca': True,
      'options': {
        'weighted_by_score': True
      }
    })}
  ]

  # Setup the runner
  runner = ActiveLearningComparison(dset.train,
                                    dset.test,
                                    ResNet20,
                                    # mobilenet_wrapper_factory(models.mobilenet_v2, 10),
                                    optim.SGD,
                                    epochs=70,
                                    learning_rate=0.03,
                                    query_percent=0.1,
                                    seed_percent=0.1,
                                    query_types=query_types,
                                    scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                                    scheduler_type='train_acc',
                                    # initial_class_sample=200,
                                    batch_size=128,
                                    log_freq=10,
                                    log_level=2,
                                    run_id=500002,
                                    load_from_another_seed=None)

  print("Initialised models.")
  
  query_iterations = 3
  for i in range(query_iterations):
    print(f'Iteration: {runner.train_iter}')
    runner.run_train_and_query()
  
  # runner.run_validation(iterations=5, log_freq=1, log_level=2, epochs=90)
  
  # plot = Plotter()
  # plt = plot.get_plots_from_file(runner.q_names, runner.save_loc + '/val-results0.json')
  # plt.show()

  # plot = Plotter()
  # plt = plot.get_plots_from_multiple_files_joined(runner.q_names, [
  #           runner.save_loc + '/val-results0.json', 
  #           runner.save_loc + '/val-results1.json'])
  # plt.show()

  # plot = Plotter()
  # plt = plot.get_plots_from_multiple_files([['random', 'confidence', 'k-nearest-confidence'], 
  #                                              ['k-nearest-confidence-mix-15-n', 'k-nearest-confidence-mix-30-n', 'confidence-mix'],
  #                                              ['k-center'],
  #                                              ['k-center-mix']], 
  #                                             ['./results/runs/300007'+'/val-results.json', 
  #                                              './results/runs/300003'+'/val-results.json',
  #                                              './results/runs/300010'+'/val-results.json',
  #                                              './results/runs/300011'+'/val-results.json'], eps_interval=1)
  # plt.show()

  # tsne_plot = TSNEPlotter([
  #                           'random',
  #                           'confidence',
  #                           'k-nearest-confidence',
  #                           'k-center',
  #                           'k-15-nearest-confidence-mix',
  #                           'k-15-nearest-confidence-mix',
  #                           'confidence-mix',
  #                           'k-center-mix'
  #                         ], 
  #                         './results/runs/400004/idx_4.npz', 
  #                         './results/runs/400004/idx_3.npz', 
  #                         [
  #                           './results/runs/400004/model_random_iter_4.pt',
  #                           './results/runs/400004/model_confidence_iter_4.pt',
  #                           './results/runs/400004/model_k-nearest-confidence_iter_4.pt',
  #                           './results/runs/400004/model_k-center-mix_iter_4.pt',
  #                           './results/runs/400004/model_k-nearest-confidence-mix-15-n_iter_4.pt',
  #                           './results/runs/400004/model_k-nearest-confidence-mix-30-n_iter_4.pt',
  #                           './results/runs/400004/model_k-center-mix_iter_4.pt',
  #                         ], 
  #                         ResNet20,
  #                         # mobilenet_wrapper_factory(models.mobilenet_v2, 3), 
  #                         dset.query)
  # tsne_plot.generate_tsne(idx=[0, 3, 6])