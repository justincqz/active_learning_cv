from al_sampling.sample_factory import ActiveLearningSampler

def use_predefined_experiments(options, batch_size=128):
  query_types = [
      {'name': 'margin-weighted-coreset-cosine', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.margin,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'options': {
          'weighted_by_score': True,
          'distance_metric': 'cosine'
        }
      })},
      {'name': 'margin-weighted-coreset-cosine-pca', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.margin,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'use_pca': True,
        'options': {
          'weighted_by_score': True,
          'distance_metric': 'cosine'
        }
      })},
      {'name': 'conf-weighted-coreset-cosine', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.confidence,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'options': {
          'weighted_by_score': True,
          'distance_metric': 'cosine'
        }
      })},
      {'name': 'conf-weighted-coreset-cosine-pca', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.confidence,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'use_pca': True,
        'options': {
          'weighted_by_score': True,
          'distance_metric': 'cosine'
        }
      })},
      {'name': 'grad-weighted-coreset-cosine', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.gradients,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'use_pca': True,
        'options': {
          'weighted_by_score': True,
          'distance_metric': 'cosine'
        }
      })},
      {'name': 'grad-weighted-coreset-cosine-pca', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.gradients,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'use_pca': True,
        'options': {
          'weighted_by_score': True,
          'distance_metric': 'cosine'
        }
      })},
      {'name': 'margin-weighted-coreset-euclidean', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.margin,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'options': {
          'weighted_by_score': True
        }
      })},
      {'name': 'margin-weighted-coreset-euclidean-pca', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.margin,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'use_pca': True,
        'options': {
          'weighted_by_score': True
        }
      })},
      {'name': 'grad-weighted-coreset-euclidean', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.gradients,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'options': {
          'weighted_by_score': True
        }
      })},
      {'name': 'grad-weighted-coreset-euclidean-pca', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.gradients,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'use_pca': True,
        'options': {
          'weighted_by_score': True
        }
      })},
      {'name': 'grad-weighted-coreset-manhattan', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.gradients,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'options': {
          'weighted_by_score': True,
          'distance_metric': 'manhattan'
        }
      })},
      {'name': 'grad-weighted-coreset-manhattan-pca', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.gradients,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'use_pca': True,
        'options': {
          'weighted_by_score': True,
          'distance_metric': 'manhattan'
        }
      })},
      {'name': 'margin-weighted-coreset-manhattan', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.margin,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'options': {
          'weighted_by_score': True,
          'distance_metric': 'manhattan'
        }
      })},
      {'name': 'margin-weighted-coreset-manhattan-pca', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.margin,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'use_pca': True,
        'options': {
          'weighted_by_score': True,
          'distance_metric': 'manhattan'
        }
      })},
      {'name': 'conf-weighted-coreset-euclidean', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.confidence,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'options': {
          'weighted_by_score': True
        }
      })},
      {'name': 'conf-weighted-coreset-euclidean-pca', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.confidence,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'use_pca': True,
        'options': {
          'weighted_by_score': True
        }
      })},
      {'name': 'conf-weighted-coreset-manhattan', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.confidence,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'use_pca': True,
        'options': {
          'weighted_by_score': True,
          'distance_metric': 'manhattan'
        }
      })},
      {'name': 'conf-weighted-coreset-manhattan-pca', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.confidence,
        'diversity': ActiveLearningSampler.diversity.coreset,
        'batch_size': batch_size,
        'use_pca': True,
        'options': {
          'weighted_by_score': True,
          'distance_metric': 'manhattan'
        }
      })},
      {'name': 'random', 'func': ActiveLearningSampler({
        'entropy': ActiveLearningSampler.entropy.confidence,
        'diversity': ActiveLearningSampler.diversity.random,
        'diversity_mix': 0.0,
        'passive_learning': 1.0,
        'batch_size': batch_size
      })},
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

  if hasattr(options, '__iter__'):
    new_types = []
    for i in options:
      new_types.append(query_types[i])
    return new_types
  elif options == 1:
    return query_types[:18]
  elif options == 2:
    return query_types[18:]
  else:
    return []