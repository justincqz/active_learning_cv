# Flags
from training.manager import ActiveLearningComparison
from training.plotter import Plotter, TSNEPlotter
from training.utils import output_wrapper_factory, mobilenet_wrapper_factory
from data_processing.cifar10 import Cifar10DS
from data_processing.dog import DogDS
from data_processing.covid import CovidDS
from al_sampling.sample_factory import ActiveLearningSampler
from constants import ConfigManager

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

  query_types = [
    # {'name': 'margin-weighted-coreset-cosine', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.margin,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'options': {
    #     'weighted_by_score': True,
    #     'distance_metric': 'cosine'
    #   }
    # })},
    # {'name': 'margin-weighted-coreset-cosine-pca', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.margin,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'use_pca': True,
    #   'options': {
    #     'weighted_by_score': True,
    #     'distance_metric': 'cosine'
    #   }
    # })},
    # {'name': 'conf-weighted-coreset-cosine', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.confidence,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'options': {
    #     'weighted_by_score': True,
    #     'distance_metric': 'cosine'
    #   }
    # })},
    # {'name': 'conf-weighted-coreset-cosine-pca', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.confidence,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'use_pca': True,
    #   'options': {
    #     'weighted_by_score': True,
    #     'distance_metric': 'cosine'
    #   }
    # })},
    # {'name': 'grad-weighted-coreset-cosine', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.gradients,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'use_pca': True,
    #   'options': {
    #     'weighted_by_score': True,
    #     'distance_metric': 'cosine'
    #   }
    # })},
    # {'name': 'grad-weighted-coreset-cosine-pca', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.gradients,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'use_pca': True,
    #   'options': {
    #     'weighted_by_score': True,
    #     'distance_metric': 'cosine'
    #   }
    # })},
    # {'name': 'margin-weighted-coreset-euclidean', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.margin,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'options': {
    #     'weighted_by_score': True
    #   }
    # })},
    # {'name': 'margin-weighted-coreset-euclidean-pca', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.margin,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'use_pca': True,
    #   'options': {
    #     'weighted_by_score': True
    #   }
    # })},
    # {'name': 'grad-weighted-coreset-euclidean', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.gradients,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'options': {
    #     'weighted_by_score': True
    #   }
    # })},
    # {'name': 'grad-weighted-coreset-euclidean-pca', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.gradients,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'use_pca': True,
    #   'options': {
    #     'weighted_by_score': True
    #   }
    # })},
    # {'name': 'grad-weighted-coreset-manhattan', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.gradients,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'options': {
    #     'weighted_by_score': True,
    #     'distance_metric': 'manhattan'
    #   }
    # })},
    # {'name': 'grad-weighted-coreset-manhattan-pca', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.gradients,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'use_pca': True,
    #   'options': {
    #     'weighted_by_score': True,
    #     'distance_metric': 'manhattan'
    #   }
    # })},
    # {'name': 'margin-weighted-coreset-manhattan', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.margin,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'options': {
    #     'weighted_by_score': True,
    #     'distance_metric': 'manhattan'
    #   }
    # })},
    # {'name': 'margin-weighted-coreset-manhattan-pca', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.margin,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'use_pca': True,
    #   'options': {
    #     'weighted_by_score': True,
    #     'distance_metric': 'manhattan'
    #   }
    # })},
    # {'name': 'conf-weighted-coreset-euclidean', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.confidence,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'options': {
    #     'weighted_by_score': True
    #   }
    # })},
    # {'name': 'conf-weighted-coreset-euclidean-pca', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.confidence,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'use_pca': True,
    #   'options': {
    #     'weighted_by_score': True
    #   }
    # })},
    # {'name': 'conf-weighted-coreset-manhattan', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.confidence,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'use_pca': True,
    #   'options': {
    #     'weighted_by_score': True,
    #     'distance_metric': 'manhattan'
    #   }
    # })},
    # {'name': 'conf-weighted-coreset-manhattan-pca', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.confidence,
    #   'diversity': ActiveLearningSampler.diversity.coreset,
    #   'batch_size': batch_size,
    #   'use_pca': True,
    #   'options': {
    #     'weighted_by_score': True,
    #     'distance_metric': 'manhattan'
    #   }
    # })},
    # {'name': 'random', 'func': ActiveLearningSampler({
    #   'entropy': ActiveLearningSampler.entropy.confidence,
    #   'diversity': ActiveLearningSampler.diversity.random,
    #   'diversity_mix': 0.0,
    #   'passive_learning': 1.0,
    #   'batch_size': batch_size
    # })},
    {'name': 'margin', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.margin,
      'diversity': ActiveLearningSampler.diversity.random,
      'diversity_mix': 0.0,
      'batch_size': batch_size
    })},
    {'name': 'margin-mix', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.margin,
      'diversity': ActiveLearningSampler.diversity.random,
      'diversity_mix': 0.2,
      'batch_size': batch_size
    })},
    {'name': 'k-nearest-margin', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.margin,
      'diversity': ActiveLearningSampler.diversity.knearest,
      'batch_size': batch_size
    })},
    {'name': 'k-nearest-margin-mix', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.margin,
      'diversity': ActiveLearningSampler.diversity.knearest,
      'passive_learning': 0.2,
      'batch_size': batch_size
    })},
    {'name': 'k-center-margin', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.margin,
      'diversity': ActiveLearningSampler.diversity.coreset,
      'batch_size': batch_size,
      'options': {
        'weighted_by_score': True,
        'distance_metric': 'euclidean'
      }
    })},
    {'name': 'k-center-margin-mix', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.margin,
      'diversity': ActiveLearningSampler.diversity.coreset,
      'passive_learning': 0.2,
      'batch_size': batch_size,
      'options': {
        'weighted_by_score': True,
        'distance_metric': 'euclidean'
      }
    })},
    {'name': 'conf', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.confidence,
      'diversity': ActiveLearningSampler.diversity.random,
      'diversity_mix': 0.0,
      'batch_size': batch_size
    })},
    {'name': 'conf-mix', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.confidence,
      'diversity': ActiveLearningSampler.diversity.random,
      'diversity_mix': 0.2,
      'batch_size': batch_size
    })},
    {'name': 'k-nearest-conf', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.confidence,
      'diversity': ActiveLearningSampler.diversity.knearest,
      'batch_size': batch_size
    })},
    {'name': 'k-nearest-conf-mix', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.confidence,
      'diversity': ActiveLearningSampler.diversity.knearest,
      'passive_learning': 0.2,
      'batch_size': batch_size
    })},
    {'name': 'k-center-conf', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.confidence,
      'diversity': ActiveLearningSampler.diversity.coreset,
      'batch_size': batch_size,
      'options': {
        'weighted_by_score': True,
        'distance_metric': 'euclidean'
      }
    })},
    {'name': 'k-center-conf-mix', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.confidence,
      'diversity': ActiveLearningSampler.diversity.coreset,
      'passive_learning': 0.2,
      'batch_size': batch_size,
      'options': {
        'weighted_by_score': True,
        'distance_metric': 'euclidean'
      }
    })},
    {'name': 'grad', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.gradients,
      'diversity': ActiveLearningSampler.diversity.random,
      'diversity_mix': 0.0,
      'batch_size': batch_size
    })},
    {'name': 'grad-mix', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.gradients,
      'diversity': ActiveLearningSampler.diversity.random,
      'diversity_mix': 0.2,
      'batch_size': batch_size
    })},
    {'name': 'k-nearest-grad', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.gradients,
      'diversity': ActiveLearningSampler.diversity.knearest,
      'batch_size': batch_size
    })},
    {'name': 'k-nearest-grad-mix', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.gradients,
      'diversity': ActiveLearningSampler.diversity.knearest,
      'passive_learning': 0.2,
      'batch_size': batch_size
    })},
    {'name': 'k-center-grad', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.gradients,
      'diversity': ActiveLearningSampler.diversity.coreset,
      'batch_size': batch_size,
      'options': {
        'weighted_by_score': True,
        'distance_metric': 'euclidean'
      }
    })},
    {'name': 'k-center-grad-mix', 'func': ActiveLearningSampler({
      'entropy': ActiveLearningSampler.entropy.gradients,
      'diversity': ActiveLearningSampler.diversity.coreset,
      'passive_learning': 0.2,
      'batch_size': batch_size,
      'options': {
        'weighted_by_score': True,
        'distance_metric': 'euclidean'
      }
    })},
  ]

  # Setup the runner
  runner = ActiveLearningComparison(dset.train,
                                    dset.test,
                                    # ResNet20,
                                    mobilenet_wrapper_factory(models.mobilenet_v2, 10, intermediate_dim=320),
                                    optim.SGD,
                                    # epochs=70,
                                    epochs=20,
                                    # learning_rate=0.03,
                                    learning_rate=0.02,
                                    query_percent=0.1,
                                    seed_percent=0.1,
                                    query_types=query_types,
                                    scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                                    scheduler_type='train_acc',
                                    # initial_class_sample=200,
                                    batch_size=batch_size,
                                    log_freq=2,
                                    log_level=2,
                                    run_id=800002,
                                    load_from_another_seed=None)

  print("Initialised models.")
  
  query_iterations = 3
  for i in range(query_iterations):
    print(f'Iteration: {runner.train_iter}')
    runner.run_train_and_query()
  
  runner.run_validation(iterations=5, log_freq=2, log_level=2, epochs=20, log_start=12)

  # print(f'Iteration: {runner.train_iter}')
  # runner.run_train_and_query()

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