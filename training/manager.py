from al_sampling.utils import initialise_seed_dataloader, create_dataloader_from_indices
from training.train import train_part
from training.utils import log_timestamp, check_accuracy, save_model
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
from shutil import copyfile

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Attempt active learning comparison
class ActiveLearningComparison():
  def __init__(self, data, test_data, model, optim, epochs=10,
               learning_rate=0.01, query_percent=0.1, seed_percent=0.1, 
               query_types=[], random_seed=42, scheduler=None, scheduler_type=None, 
               loss_func=None, initial_class_sample=0, batch_size=128,
               verbose=True, log_freq=1, log_level=2, run_id=None, load_from_another_seed=None):
    
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
    self.scheduler_type = scheduler_type if scheduler_type is not None else 'train_acc'
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

    self.to_train = [] # List indicating which indicies of models to train, used for only training certain models based on progress, first run etc
    self.cur_idx = []
    self.train_loaders = []
    self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size)
    self.results = []
    
    # If given a run_id and Google Drive is mounted, check if we can load some current progress
    loaded = False
    if not (run_id is None or SAVE_LOC is None):
      
      # Load from another seed to create coherent results
      if load_from_another_seed is not None:
        self.copy_over_seed(SAVE_LOC+'/runs', run_id, load_from_another_seed)

      loaded = self.load_run_id(SAVE_LOC+'/runs', run_id, prompt=True)

    # If Google Drive is mounted but we failed to load any data, create the files in Drive
    if not (SAVE_LOC is None or loaded):
      self.create_run_id(SAVE_LOC+'/runs', run_id)

    # Initialise seed loaders if we are not continuing from before
    if not loaded:
      self.create_indicies()
      self.save_indices()

    self.init_models(loaded=loaded)

    # Set seed for reproducibility
    torch.manual_seed(self.r_seed)
    np.random.seed(self.r_seed)

  def create_indicies(self):
    assert len(self.cur_idx) == 0
    seed_loader, known_index = initialise_seed_dataloader(self.data, self.s_percent, min_samples=self.initial_class_sample, batch_size=self.batch_size)
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
  
  def copy_over_seed(self, location, run_id, other_run):
    run_id = '%.6d' % run_id
    other_run = '%.6d' % other_run
    
    if os.path.isfile(location+f'/{run_id}/run_details.json'):
      if self.verbose:
        print('Run details already initialised, proceed with loading as normal.')
      return False
    
    # File existence check
    if not os.path.isfile(location+f'/{other_run}/idx_0.npz'):
      if self.verbose:
        print(f"Failed t copy over seed frm run {run_id}, can't find its seed file (idx_0.npz).")
      return False
    
    # Read the seed of the other run and replicate the seed with the correct number of queries
    index_zip = np.load(location+f'/{other_run}/idx_0.npz')
    idx_map = { str(i):index_zip["0"] for i in range(len(self.q_types)) }
    os.makedirs(location+f'/{run_id}', mode=711, exist_ok=True)
    np.savez(location+f'/{run_id}/idx_0', **idx_map)
    np.savez(location+f'/{run_id}/idx', **idx_map)                   # This is the main copy used for loading
    
    # We also create the metadata to fool the loader into thinking that this is a loadable instance
    metadata = self.get_metadata(run_id)
    f = open(location+f'/{run_id}/run_details.json','w')
    f.write(str(metadata))
    f.close()
   
    
    if not os.path.isfile(location+f'/{run_id}/idx.npz'):
      if self.verbose:
        print(f"Failed to copy over seed from run {other_run}, couldn't copy the file over.")
      return False
    
    if self.verbose:
      print(f'Successfully copied over seed from run {other_run}.')
      
    return True
    
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

  def save_model(self, model_idx: int, filename: str) -> None:
    if self.save_loc is None:
      return

    torch.save(self.models[model_idx].cpu().state_dict(), self.save_loc+f'/model_{filename}_iter_{self.train_iter}.pt')
    if self.verbose:
      print(f'Saved trained model to {self.save_loc}/model_{filename}_iter_{self.train_iter}.pt')
    
    self.models[model_idx].to(device)

  def load_model(self, location, to_device=True):
    assert os.path.isfile(location)
    try:
      m = self.model()
      state_dict = torch.load(location)
      m.load_state_dict(state_dict)
      if to_device:
        m.to(device)
    except Exception:
      if self.verbose:
        print(f'Unable to load trained model. \n Location: {location}')
      return None
    
    return m
  
  def load_run_id(self, location, run_id, load_if_possible=False, prompt=False):
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
    
    # Update current parameters with the saved run progress
    # Read the results file if it exists (not the end of the world if it doesn't exist)
    # If we saved the model of a certain train iteration, test it and ask the user if we want to load it
    if os.path.isfile(self.save_loc+f'/results.json'):
      f = open(self.save_loc+f'/results.json','r')
      res_data = ''.join(f.readlines())
      f.close()
      res_data = literal_eval(res_data)
      self.train_iter = res_data['iter']
      self.results = list(res_data['results'].values())
      self.done_query = res_data['done_query']
      self.to_train = []
      self.models = [None] * len(self.q_types)
      
      if load_if_possible or prompt:
        first_prompt = False
        want_models = True
        for i in range(len(self.q_names)):
          if os.path.isfile(f'{self.save_loc}/model_{self.q_names[i]}_iter_{self.train_iter}.pt') and want_models:
            if prompt and not first_prompt:
              want_models = ask_bool_prompt('Found models that can be loaded for querying. Do you want to load these models? [Y/N]:')
              first_prompt = True
              if not want_models:
                self.to_train.append(i)
                continue
            
            m = self.load_model(f'{self.save_loc}/model_{self.q_names[i]}_iter_{self.train_iter}.pt')
            if m is None:
              self.to_train.append(i)
              continue
            if not load_if_possible:
              test_acc = check_accuracy(self.test_loader, m, verbose=False)
              allow = ask_bool_prompt(f'Loaded model {self.q_names[i]} with test accuracy of {test_acc}, do you want to load this result and directly query the model? [Y/N]:')
              if not allow:
                self.to_train.append(i)
                continue

            self.models[i] = m
          else:
            self.to_train.append(i)
    else:
      self.results = [[] for _ in range(len(self.q_types))]
      self.to_train = [i for i in range(len(self.q_types))]

    # Check if the indices file is loaded (important as without indicies, nothing to load, generate new ones instead)
    if not os.path.isfile(self.save_loc+f'/idx.npz'):
      self.create_indicies()
      self.run_id = run_id
      return True

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

  def init_models(self, loaded=False):
    
    if not loaded or not hasattr(self, 'models') or len(self.models) != len(self.q_types):
      self.to_train = [i for i in range(len(self.q_types))]
      self.models   = [None] * len(self.q_types)
    self.optims     = [None] * len(self.q_types)
    self.schedulers = [None] * len(self.q_types)

    missing_models = [i for i in range(len(self.models)) if self.models[i] is None]
    missing_optims = [i for i in range(len(self.optims)) if self.optims[i] is None]
    missing_schedulers = [i for i in range(len(self.schedulers)) if self.schedulers[i] is None]
    for i in range(len(missing_models)):
      self.models[i] = self.model().to(device)
    
    for i in range(len(missing_optims)):
      self.optims[i] = self.optim(self.models[i].parameters(), lr=self.lr)
    
    for i in range(len(missing_schedulers)):
      self.schedulers[i] = self.scheduler(self.optims[i], mode='max', factor=0.5, patience=1)
    
    assert len([0 for i in self.models if i is None]) == 0
    assert len([0 for i in self.optims if i is None]) == 0
    assert len([0 for i in self.schedulers if i is None]) == 0

  def run_train(self, save=True, first_run=False, save_models=True):
    if first_run:
      self.to_train = [0]
      
    res = None

    # Train each model
    for i in self.to_train:
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
                  log_level=self.log_level,
                  save_best_model=save and save_models,
                  save_path=self.save_loc+f'/model_{self.q_names[i]}_iter_{self.train_iter}.pt'
                  )
      
      log_timestamp()

      if save:
        self.results[i].append(res)
        self.save_results()

      if save_models:
        self.models[i] = self.load_model(self.save_loc+f'/model_{self.q_names[i]}_iter_{self.train_iter}.pt')

      if first_run:
        if save_models:
          self.models[0] = self.load_model(self.save_loc+f'/model_{self.q_names[0]}_iter_{self.train_iter}.pt')
        for k in range(1, len(self.models)):
          if save:
            self.results[k].append(res)
          if save_models:
            copyfile(self.save_loc+f'/model_{self.q_names[0]}_iter_{self.train_iter}.pt', 
                     self.save_loc+f'/model_{self.q_names[k]}_iter_{self.train_iter}.pt')
          self.models[k] = copy.deepcopy(self.models[0])

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
        temp_df['class'] = np.array(self.data.classes)[[int(k) for k in temp_df.index]]
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
    self.train_iter += 1

  def run_train_and_query(self, save_models=True):
    is_first_run = self.copy_on_first_iter and self.train_iter == 0
    self.run_train(save=True, first_run=is_first_run, save_models=save_models)
    self.run_query()

  def run_validation(self, iterations=3, log_freq=None, log_level=None, epochs=None, log_start=0, only_run=None):
    results = [[] for _ in range(len(self.q_types))]
    l_freq  = log_freq  if not log_freq  is None else self.log_freq
    l_level = log_level if not log_level is None else self.log_level
    eps     = epochs    if not epochs    is None else self.epochs
    to_run  = range(len(self.models)) if only_run is None else only_run

    # Make sure that each validation run is not overidden
    filename = 'val-results'
    count = 0
    found_file = os.path.isfile(self.save_loc+f'/{filename}.json')
    while found_file:
      if os.path.isfile(self.save_loc+f'/{filename}{count}.json'):
        count += 1
      else:
        filename = f'{filename}{count}'
        break

    for it in range(iterations):
      # Initialise and reset model, optimisers and schedulers
      self.init_models()
      
      if self.verbose:
        print(f'Validation iteration: {it}')
        log_timestamp()

      for i in to_run:
        if self.verbose:
          print("| Training %s |" % (self.q_names[i]))
          log_timestamp()

        # Tensorboard Logging
        if USE_TB:
          loss_w = SummaryWriter(f'{LOG_DIR}/{self.run_id}/{self.q_names[i]}/val-iter-{it}/loss')
          train_w = SummaryWriter(f'{LOG_DIR}/{self.run_id}/{self.q_names[i]}/val-iter-{it}/train')
          test_w = SummaryWriter(f'{LOG_DIR}/{self.run_id}/{self.q_names[i]}/val-iter-{it}/test')
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
                  log_level=l_level,
                  log_start=log_start)

        results[i].append(res)
        self.save_results(filename=filename, results=results)

      
    # Reset the models
    self.init_models()

    return res

def ask_bool_prompt(msg: str) -> bool:
  ans = input(msg+' ')
  while ans.lower() not in ['y', 'n', 'yes', 'no']:
    print('That input could not be recognised. Accepted inputs are [Y, yes, y] and [N, n, no].')
    ans = input(msg+' ')
  return ans.lower() in ['y', 'yes']