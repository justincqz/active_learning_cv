from al_sampling.utils import initialise_seed_dataloader, create_dataloader_from_indices
from training.train import train_part
from training.utils import log_timestamp
from constants import ConfigManager

USE_TB = ConfigManager.USE_TB
SAVE_LOC = ConfigManager.SAVE_LOC
LOG_DIR = ConfigManager.LOG_DIR
device = ConfigManager.device

if USE_TB:
  from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from ast import literal_eval

import math
import os
import copy

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Attempt active learning comparison
class ActiveLearningComparison():
  def __init__(self, data, test_data, model, optim, epochs=10,
               learning_rate=0.01, query_percent=0.1, seed_percent=0.1, 
               query_types=[], random_seed=42, scheduler=None, scheduler_type='epoch', 
               loss_func=None, initial_class_sample=0, batch_size=128,
               verbose=True, log_freq=1, log_level=2, run_id=None):
    
    self.ds_name = data.root.split('/')[-1]
    self.data = data
    self.q_names = [q['name'] for q in query_types]
    self.s_percent = seed_percent
    self.test_data = test_data
    self.epochs = epochs
    self.train_iter = 0
    self.lr = learning_rate
    self.q_size = math.floor(len(data.data) * query_percent)
    self.q_types = [q['func'] for q in query_types]
    self.r_seed = random_seed
    self.scheduler = scheduler
    self.scheduler_type = scheduler_type
    self.model = model
    self.optim = optim
    self.loss_fs = loss_func
    self.batch_size = batch_size
    self.verbose = verbose
    self.copy_on_first_iter = True
    self.log_freq = log_freq
    self.log_level = log_level
    self.initial_class_sample = initial_class_sample
    self.done_query = False

    self.cur_idx = []
    self.train_loaders = []
    self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, pin_memory=True)
    self.results = []

    # If given a run_id and Google Drive is mounted, check if we can load some current progress
    loaded = False
    if not (run_id is None or SAVE_LOC is None):
      loaded = self.load_run_id(SAVE_LOC+'/runs', run_id)

    # If Google Drive is mounted but we failed to load any data, create the files in Drive
    if not (SAVE_LOC is None or loaded):
      self.create_run_id(SAVE_LOC+'/runs', run_id)

    # Initialise seed loaders if we are not continuing from before
    if not loaded:
      self.create_indicies()

    self.init_models()

    # Set seed for reproducibility
    torch.manual_seed(self.r_seed)
    np.random.seed(self.r_seed)

  def create_indicies(self):
    assert len(self.cur_idx) == 0
    seed_loader, known_index = initialise_seed_dataloader(self.data, 
                                                          self.s_percent, 
                                                          min_samples=self.initial_class_sample, 
                                                          batch_size=self.batch_size)
    for _ in range(len(self.q_types)):
      self.train_loaders.append(seed_loader)
      self.cur_idx.append(known_index)
      self.results.append([])

    self.save_progress()
  
  def get_metadata(self, run_id):
    metadata = {
        'run_id': run_id, 
        'query_types': self.q_names, 
        'model': self.model.__name__, 
        'dataset': self.ds_name,
        'seed_percent': self.s_percent
    }
    return metadata

  def create_run_id(self, location, run_id=None):
    if run_id is None:
      found_new_id = False
      while not found_new_id:
        run_id = '%.6d' % (np.random.randint(0, 1000000),)
        # File existence check
        if not os.path.isfile(location+f'/{run_id}/run_details.json'):
          break
    else:
      if os.path.isfile(location+f'/{run_id}/run_details.json'):
        raise FileExistsError('Unable to create run at selected id, directory already exists.')

    run_id = '%.6d' % run_id if type(run_id) == int else run_id
    os.makedirs(location+f'/{run_id}', mode=711, exist_ok=True)
    # Save the metadata to file
    metadata = self.get_metadata(run_id)
    f = open(location+f'/{run_id}/run_details.json','w')
    f.write(str(metadata))
    f.close()
    self.save_loc = location+f'/{run_id}'
    self.run_id = run_id
    return

  def save_progress(self):
    self.save_indices()
    self.save_results()

  def save_indices(self):
    if self.save_loc is None or len(self.cur_idx) < 1:
      return

    assert len(self.cur_idx) == len(self.q_types)
    idx_map = { str(i):self.cur_idx[i] for i in range(len(self.q_types)) }
    np.savez(self.save_loc+f'/idx_{self.train_iter}', **idx_map) # Save this copy for logging purposes
    np.savez(self.save_loc+f'/idx', **idx_map)                   # This is the main copy used for loading

    if self.verbose:
      print(f'Saved indices to {self.save_loc}/idx_{self.train_iter}.npz')

  def save_results(self, filename='results', results=None):
    results = self.results if results is None else results
    if self.save_loc is None or results is None:
      return

    res_dict = {i:results[i] for i in range(len(results))}
    data = { 'iter': self.train_iter, 'results': res_dict, 'done_query': self.done_query }
    f = open(self.save_loc+f'/{filename}.json','w')
    f.write(str(data))
    f.close()

    if self.verbose:
      print(f'Saved results to {self.save_loc}/{filename}.json')

  def save_model(self, model_idx, filename):
    if self.save_loc is None:
      return

    torch.save(self.models[model_idx].cpu().state_dict(), self.save_loc+f'/model_{filename}_iter_{self.train_iter}.pt')
    if self.verbose:
      print(f'Saved trained model to {self.save_loc}/model_{filename}_iter_{self.train_iter}.pt')
    
    self.models[model_idx].to(device)

  
  def load_run_id(self, location, run_id):
    run_id = '%.6d' % run_id
    # File existence check
    if not os.path.isfile(location+f'/{run_id}/run_details.json'):
      return False

    # Read the metadata file
    f = open(location+f'/{run_id}/run_details.json','r')
    metadata = ''.join(f.readlines())
    f.close()
    metadata = literal_eval(metadata)

    # Check if the parameters is identical to the metadata
    metadata_new = self.get_metadata(run_id)
    if not metadata == metadata_new:
      print(metadata)
      print(metadata_new)
      raise KeyError('Metadata is different, choose another run id')
    
    self.save_loc = location+f'/{run_id}'

    # Check if the indices file is loaded (important as without indicies, nothing to load, generate new ones instead)
    if not os.path.isfile(self.save_loc+f'/idx.npz'):
      self.create_indicies()
      self.run_id = run_id
      return True

    # Update current parameters with the saved run progress
    # Read the results file if it exists (not the end of the world if it doesn't exist)
    if os.path.isfile(self.save_loc+f'/results.json'):
      f = open(self.save_loc+f'/results.json','r')
      res_data = ''.join(f.readlines())
      f.close()
      res_data = literal_eval(res_data)
      self.train_iter = res_data['iter']
      self.results = list(res_data['results'].values())
      self.done_query = res_data['done_query']
    else:
      self.results = [[] for _ in range(len(self.q_types))]

    # Read the indicies file
    index_zip = np.load(self.save_loc+f'/idx.npz')
    self.cur_idx = [index_zip[str(i)] for i in range(len(self.q_types))]
    self.train_loaders = [create_dataloader_from_indices(self.data, 
                                                         self.cur_idx[i], 
                                                         batch_size=self.batch_size) for i in range(len(self.q_types))]

    if self.verbose:
      print(f'Successfully loaded previous run and indicies at iteration {self.train_iter}.')

    self.run_id = run_id
    return True

  def init_models(self):
    self.models = []
    self.optims = []
    self.schedulers = []

    for _ in range(len(self.q_types)):
      m = self.model().to(device)
      op = self.optim(m.parameters(), lr=self.lr)
      s = self.scheduler(op, mode='max', factor=0.5, patience=1)
      self.models.append(m)
      self.optims.append(op)
      self.schedulers.append(s)

  def run_train(self, save=True, first_run=False):
    self.train_iter += 1
    train_count = 1 if first_run else len(self.models)
    
    # Train each model
    for i in range(train_count):
      if self.verbose:
        if first_run:
          print("| Running first training session with a single model and copying |")
        else:
          print("| Training %s |" % (self.q_names[i]))
        log_timestamp()
      
      # Tensorboard Logging
      if USE_TB and not first_run:
        loss_w = SummaryWriter(f'{LOG_DIR}/{self.run_id}/{self.q_names[i]}/iter-{self.train_iter}/loss')
        train_w = SummaryWriter(f'{LOG_DIR}/{self.run_id}/{self.q_names[i]}/iter-{self.train_iter}/train')
        test_w = SummaryWriter(f'{LOG_DIR}/{self.run_id}/{self.q_names[i]}/iter-{self.train_iter}/test')
        writer = (loss_w, train_w, test_w)
      else:
        writer = None

      res = train_part(self.models[i],
                  optimizer=self.optims[i], 
                  epochs=self.epochs, 
                  train_data=self.train_loaders[i], 
                  test_data=self.test_loader,
                  writer=writer,
                  scheduler=self.schedulers[i],
                  scheduler_type=self.scheduler_type,
                  log_freq=self.log_freq,
                  log_level=self.log_level
                  )
      
      log_timestamp()
      if first_run:
        for k in range(1, len(self.models)):
          self.models[k] = copy.deepcopy(self.models[0])
          if save:
            self.results[k].append(res)

      if save:
        self.results[i].append(res)
        self.save_results()
        self.save_model(i, self.q_names[i])
        
    return res

  def run_query(self):
    if self.q_size < 1:
      return

    if self.verbose:
      print("| Adding %d new datapoints |" % self.q_size)

    for i in range(len(self.q_types)):
      if self.verbose:
        print("Querying using %s" % (self.q_names[i]))
        log_timestamp()

      if USE_TB:
        writer = SummaryWriter(f'{LOG_DIR}/{self.run_id}/{self.q_names[i]}/iter-{self.train_iter}/plot')
      else:
        writer = None

      prev_idx = self.cur_idx[i]
      self.cur_idx[i] = self.q_types[i].query(self.q_size, self.cur_idx[i], self.data, self.models[i], writer=writer)
      new_idx = prev_idx ^ np.copy(self.cur_idx[i])
      
      if self.verbose:
        print(f'Querying complete, added {new_idx.sum()} datapoints.')
        log_timestamp()

      if not writer is None:
        fig = plt.figure()
        targets = np.array(self.data.targets)[new_idx]
        temp_df = pd.DataFrame(zip(range(len(targets)), targets), columns=['idx', 'label'])
        temp_df = temp_df.groupby('label').count()
        temp_df['class'] = self.data.classes
        plt.bar(range(len(temp_df)), temp_df['idx'], align='center')
        plt.xticks(range(len(temp_df)), temp_df['class'], size='small')
        plt.title(f'New Image Class Counts for {self.q_names[i]}')
        writer.add_figure('Plot/Histogram', fig)
        
      self.train_loaders[i] = create_dataloader_from_indices(self.data, self.cur_idx[i], self.batch_size)
    
    if self.verbose:
      print("| Resetting models and optimizers |")

    # Save the indicies to Drive
    self.save_indices()

    # Reset the models
    self.init_models()

  def run_train_and_query(self):
    is_first_run = self.copy_on_first_iter and self.train_iter == 0
    self.run_train(save=True, first_run=is_first_run)
    self.run_query()

  def run_validation(self, iterations=3, log_freq=None, log_level=None, epochs=None):
    results = [[] for _ in range(len(self.q_types))]
    l_freq  = log_freq  if not log_freq  is None else self.log_freq
    l_level = log_level if not log_level is None else self.log_level
    eps     = epochs    if not epochs    is None else self.epochs

    # Make sure that each validation run is not overidden
    filename = 'val-results'
    count = 0
    found_file = os.path.isfile(self.save_loc+f'/{filename}.json')
    while found_file:
      if os.path.isfile(self.save_loc+f'/{filename}_{str(count)}.json'):
        count += 1
      else:
        filename += str(count)
        break

    for iter in range(iterations):
      # Initialise and reset model, optimisers and schedulers
      self.init_models()
      
      if self.verbose:
        print(f'Validation iteration: {iter}')
        log_timestamp()

      for i in range(len(self.models)):
        if self.verbose:
          print("| Training %s |" % (self.q_names[i]))
          log_timestamp()

        # Tensorboard Logging
        if USE_TB:
          loss_w = SummaryWriter(f'{LOG_DIR}/{self.run_id}/{self.q_names[i]}/val-iter-{self.train_iter}/loss')
          train_w = SummaryWriter(f'{LOG_DIR}/{self.run_id}/{self.q_names[i]}/val-iter-{self.train_iter}/train')
          test_w = SummaryWriter(f'{LOG_DIR}/{self.run_id}/{self.q_names[i]}/val-iter-{self.train_iter}/test')
          writer = (loss_w, train_w, test_w)
        else:
          writer = None

        res = train_part(self.models[i],
                  optimizer=self.optims[i],
                  epochs=eps,
                  train_data=self.train_loaders[i], 
                  test_data=self.test_loader,
                  writer=writer,
                  scheduler=self.schedulers[i],
                  scheduler_type=self.scheduler_type,
                  log_freq=l_freq,
                  log_level=l_level)

        results[i].append(res)
        self.save_results(filename=filename, results=results)
    
    return res