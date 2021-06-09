# Active Learning for Computer Vision
Active learning exploration for computer vision tasks, using Pytorch.

![active_learning_cycle](https://user-images.githubusercontent.com/35500556/121273871-229eae00-c8c1-11eb-8891-c69bb3bcec78.png)


This repository allows users to run an active learning training loop using their own datasets, models, hyperparameters, and active learning algorithms. There are several hard and soft pre-requisites, which include:
```python
# Hard pre-requisites
torch==1.8.1
torchvision==0.9.1
pandas==1.2.4
numpy==1.20.2
scipy==1.6.2
matplotlib==3.4.1
Pillow==8.2.0
scikit_learn==0.24.2

# Soft pre-requisites for logging and visualisation purposes
tensorboard==2.4.1
```
The main entry-point of the project is through the `main.py` file, with the centrepiece class being `ActiveLearningComparison`. 
# Running an experiment
Running an experiment is simple. First, create an experiment configuration in a python object:
```python
from training.manager import ActiveLearningComparison

experiments = [
  {'name': 'random', 'func': ActiveLearningSampler({
     'entropy': ActiveLearningSampler.entropy.confidence,
     'diversity': ActiveLearningSampler.diversity.random,
     'diversity_mix': 0.0,
     'passive_learning': 1.0
  })},
  {'name': 'k-center-grad-mix', 'func': ActiveLearningSampler({
     'entropy': ActiveLearningSampler.entropy.gradients,
     'diversity': ActiveLearningSampler.diversity.coreset,
     'passive_learning': 0.2,
     'batch_size': 128,
     'options': {
        'weighted_by_score': True,
        'distance_metric': 'euclidean'
      }
  })}]
```
Next, prepare your chosen dataset.
```
from data_processing.cifar10 import Cifar10DS
dataset = Cifar10DS()
```
Then, define your hyperparameters and initialise the `ActiveLearningComparison` module.
```python
from models import ResNet20
import torch.optim as optim

runner = ActiveLearningComparison(dset.train,
                                  dset.test,
                                  ResNet20,
                                  optim.SGD,
                                  epochs=70,
                                  learning_rate=0.03,
                                  query_percent=0.1,
                                  seed_percent=0.1,
                                  query_types=experiments,
                                  scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                                  scheduler_type='train_acc',
                                  batch_size=128,
                                  log_freq=1,
                                  log_level=2,
                                  run_id=000001,
                                  load_from_another_seed=None)
```
Finally, run a train-then-query iteration.
```python
query_iterations = 3
for i in range(query_iterations):
  print(f'Iteration: {runner.train_iter}')
  runner.run_train_and_query()
```
Run validation runs to validate your model performance:
```python
runner.run_validation(iterations=5, log_freq=1, log_level=2, epochs=70)
```

# Preparing your own data
To prepare your own dataset to use with the active learner, you have to inherit the `DataPrep` abstract class and define three key parameters, the `self.train`, `self.query`, and `self.test` parameters. All of these parameters inherit PyTorch `Dataset` objects. 

Several default datasets have been prepared in the `./data_processing` folder, which includes examples of custom pre-processing transforms being prepared for several datasets. One thing to note is that this repository does not include the actual images themselves, so the user is required to provide them in the `./data` folder. A few of the example datasets like `cifar10.py` and `fmnist.py` automatically download these images from the PyTorch CDN, and can be used immediately without any prior setup. 
# Preparing your own sampling algorithm
To prepare your own sampling algorithm, there are two methods. The first method is to utilise the `ActiveLearningSampler` and introduce your own scoring function or diversity algorithm, and take advantage of the `ActiveLearningSampler` to handle the rest. The second method is to directly inherit from the `BaseSampler` class and write your own sampler from scratch.

## Method 1: Using `ActiveLearningSampler`
Using the `ActiveLearningSampler` class to wrap around your own custom function has a number of benefits, including allowing you to mix and match with current existing `ActiveLearningSampler` compatible functions. Let's say we're trying to implement a new diversity function which uses the bottom-n scores while sampling (don't expect any great results out of this).

First, we define our diversity function:
```python
def bottom_n(query_size, known_data_indices, scores, features):
  # Initialise generic index list
  indices = np.arange(len(known_data_idx))

  # Get the bottom uncertain entries
  sorted_indices = np.argsort(-scores.cpu().numpy())

  # Add the bottom n which do not occur in the already seen indices
  new_indices = set()
  known_indices = set(indices[known_data_idx])
  for s in sorted_indices:
    # Early stopping condition
    if len(new_indices) == query_size:
      break
    # Only add unlabelled entries
    if s not in known_indices:
      new_indices.add(s)
  # Convert the set into a list for indexing later
  new_indices = list(new_indices)
  
  # Set the flags for data known
  known_data_a = known_data_idx.copy()
  known_data_a[new_indices] = True

  return known_data_a
```
Next, we need to define the experiment setup with the custom type in the diversity field, and define our function in the `diversity_function` parameter.
```python
experiments = [
  {'name': 'test-custom', 'func': ActiveLearningSampler({
     'entropy': ActiveLearningSampler.entropy.confidence,
     'diversity': ActiveLearningSampler.diversity.custom,
     'diversity_function': bottom_n
  })}]
```
That's it! If you want to allow extra optional parameters for your function, you can include it in your function definition. To define it in the experiments configuration, use the `options` field:
```python
experiments = [
  {'name': 'test-custom', 'func': ActiveLearningSampler({
     'entropy': ActiveLearningSampler.entropy.confidence,
     'diversity': ActiveLearningSampler.diversity.custom,
     'diversity_function': bottom_n,
     'options': {
        'my_custom_parameter_name': 'active_learning_cool'
      }
  })}]
```
## Method 2: Inheriting `BaseSampler`
If you want have complete control of what happens during the querying stage, you can define your own sampler and simply inherit from the `BaseSampler` class. Many earlier sampling classes have been written this way and their code still remains in the `al_sampling` folder.

# Visualising data
If you have TensorBoard installed, you can run a TensorBoard process using `tensorboard --logdir ./results/tensorboard --host 127.0.0.1 --port 6006`. The `main.py` script also automatically runs the TensorBoard subprocess in the background when starting, and you can simply comment out the `ConfigManager.load_tensorboard()` line to prevent this behaviour.

TensorBoard logs many useful metrics during training and querying, including:
Loss, Train & Test accuracy, Histogram of queried samples wrt their labels, and PR curves.

Throughout the training and validation runs, several python object log files will also be generated in the `./results/runs` folder. Each experimental run is uniquely identified by its `run_id`, and the results will also be labelled as such.

There are several tools that can help visualise validation results. 

## Plotter
The `Plotter` is used to plot average loss, train and test accuracy over each validation run, for each class. To use the `Plotter`, you can either directly utilise the current runner's configuration:
```python
plot = Plotter()
plt = plot.get_plots_from_file(runner.q_names, runner.save_loc + '/val-results.json')
plt.show()```
or simply define the query names yourself and directly load from the file:
```python
plot = Plotter()
plt = plot.get_plots_from_file(['random', 'confidence', 'k-nearest-confidence'], './results/runs/300007'+'/val-results.json')
```
The `Plotter` can also combine multiple files worth of validation results into a single plot:
```python
plt = plot.get_plots_from_multiple_files([['random', 'confidence', 'k-nearest-confidence'], 
                                          ['k-nearest-confidence-mix-15-n', 'k-nearest-confidence-mix-30-n', 'confidence-mix'],
                                          ['core-set']], 
                                         ['./results/runs/300007'+'/val-results.json', 
                                          './results/runs/300003'+'/val-results.json',
                                          './results/runs/300010'+'/val-results.json'], eps_interval=1)
```
## TSNEPlotter
The `TSNEPlotter` will load a model and dataset, run `scipy`'s TSNE function, and compute TSNE plots coloured by the classes and whether the labels are newly sampled/previously sampled/un-sampled. This is used to analyse which data instances was sampled during a query process, and can give insight into the way the model percieves the current state of the dataset. 

![random-tsne](https://user-images.githubusercontent.com/35500556/121273905-364a1480-c8c1-11eb-832e-8976fd980a99.png)


To run this, you have to define the different query names, the indices _after_ querying, the indices _before_ querying, the model, the respective model parameters used to get the feature representation, and finally the dataset. An example would look like this:
```python
tsne_plot = TSNEPlotter(['random', 'confidence', 'k-nearest-confidence'], 
                        './results/runs/400004/idx_4.npz', 
                        './results/runs/400004/idx_3.npz', 
                        ['./results/runs/400004/model_random_iter_4.pt',
                         './results/runs/400004/model_confidence_iter_4.pt',
                         './results/runs/400004/model_k-nearest-confidence_iter_4.pt'], 
                        ResNet20, 
                        dset.query)
```
To actually run the TSNE plotter, simply call the `generate_tsne` method. It accepts an optional list of integers so that you can select which query types you want to perform TSNE on, in the scenario where your experiment has too many query types and you only want TSNE plots for certain experiments.
```python
tsne_plot.generate_tsne(idx=[0, 2])
```

## Other utility functions
In the `utils` directory, there are two helper functions which can be used to help visualise certain metrics from the validation run logs.

`avg_validation_stats.py` prints out the top mean accuracy recorded, as well as the top-1 accuracy and the variance of the run for each experiment.
```
python ./utils/avg_validation_stats.py -r 800001
```

`results_parser.py` plots the top accuracy for each class after each active learning querying iteration.
```
python ./utils/results_parser.py -r 800001
```
These can be helpful to synthesise the results of the validation run and give a sense of which experiment performed the best.
