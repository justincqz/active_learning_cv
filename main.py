# Flags
from training.manager import ActiveLearningComparison
from training.plotter import Plotter, TSNEPlotter
from training.utils import output_wrapper_factory, mobilenet_wrapper_factory
from data_processing.cifar10 import Cifar10DS
from data_processing.dog import DogDS
from data_processing.covid import CovidDS
from al_sampling.sample_factory import ActiveLearningSampler
from constants import ConfigManager
from experiments import use_predefined_experiments

import torchvision.models as models
from models import ResNet20
import torch.optim as optim

if __name__ == '__main__':
  # parser = argparse.ArgumentParser(description="Active learning training environment.")
  # parser.add_argument('-lr', '--learning_rate', type=float, )

  ConfigManager.load_tensorboard()
  
  # Initialise the dataset
  dset = CovidDS()
  # dset = Cifar10DS()
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

  batch_size = 64
  # batch_size = 128

  # PREDEFINED OPTIONS:

  # 1: Feature Representation Experiment
  # Includes:
  # 0  margin-weighted-coreset-cosine, margin-weighted-coreset-cosine-pca
  # 2  conf-weighted-coreset-cosine, conf-weighted-coreset-cosine-pca
  # 4  grad-weighted-coreset-cosine, grad-weighted-coreset-cosine-pca
  # 6  margin-weighted-coreset-euclidean, margin-weighted-coreset-euclidean-pca
  # 8  grad-weighted-coreset-euclidean, grad-weighted-coreset-euclidean-pca
  # 10 grad-weighted-coreset-manhattan, grad-weighted-coreset-manhattan-pca
  # 12 margin-weighted-coreset-manhattan, margin-weighted-coreset-manhattan-pca
  # 14 conf-weighted-coreset-euclidean, conf-weighted-coreset-euclidean-pca
  # 16 conf-weighted-coreset-manhattan, conf-weighted-coreset-manhattan-pca

  # 2: Scoring function, sampling and mix experiments
  # Includes
  # 18 random
  # 19 margin, margin-mix, k-nearest-margin, k-nearest-margin-mix, k-center-margin, k-center-margin-mix
  # 25 conf, conf-mix, k-nearest-conf, k-nearest-conf-mix, k-center-conf, k-center-conf-mix
  # 31 grad, grad-mix, k-nearest-grad, k-nearest-grad-mix, k-center-grad, k-center-grad-mix
  # query_types = use_predefined_experiments(list(range(19, 37)), batch_size)
  # query_types = use_predefined_experiments([8,9,6,7,14], batch_size)
  # query_types = use_predefined_experiments([0,1,2,3,4,5,10,11,12,13,15,16,17], batch_size)
  query_types = use_predefined_experiments([18], batch_size)

  # Setup the runner
  runner = ActiveLearningComparison(dset.train,
                                    dset.test,
                                    # ResNet20,
                                    mobilenet_wrapper_factory(models.mobilenet_v2, 10, intermediate_dim=320),
                                    optim.SGD,
                                    # epochs=70,
                                    epochs=25,
                                    # learning_rate=0.03,
                                    learning_rate=0.01,
                                    query_percent=0.0,
                                    seed_percent=1.0,
                                    query_types=query_types,
                                    scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                                    scheduler_type='train_acc',
                                    # initial_class_sample=200,
                                    batch_size=batch_size,
                                    log_freq=1,
                                    log_level=2,
                                    run_id=900001,
                                    load_from_another_seed=None)

  print("Initialised models.")

  # runner.run_validation(iterations=5, log_freq=2, log_level=2, epochs=20, log_start=12)

  # query_iterations = 3
  # for i in range(query_iterations):
  #   print(f'Iteration: {runner.train_iter}')
  #   runner.run_train_and_query()
  
  # runner.run_validation(iterations=5, log_freq=1, log_level=2, epochs=20, log_start=9, only_run=[4])

  # print(f'Iteration: {runner.train_iter}')
  runner.run_train_and_query()

  # runner.run_validation(iterations=5, log_freq=1, log_level=2, epochs=20)
  
  # plot = Plotter()
  # plt = plot.get_plots_from_file(runner.q_names, runner.save_loc + '/val-results.json')
  # plt.show()

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