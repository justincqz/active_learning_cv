from ast import literal_eval
import matplotlib.pyplot as plt
import numpy as np

def get_average_values(run_id, results_files=None, verbose=False):
  run_id = '500002' if run_id is None else run_id
  target_folder = f'./results/runs/{run_id}'
  target_files = ['/val-results.json'] if results_files is None else results_files
  metadata_file = '/run_details.json'

  with open(target_folder + metadata_file, 'r') as f:
    metadata = ''.join(f.readlines())
  metadata = literal_eval(metadata)

  top_values = []
  for target_file in target_files:
    # Read JSON
    with open(target_folder + target_file, 'r') as f:
      raw = ''.join(f.readlines())
    result_dict = literal_eval(raw)

    # Get average accuracies
    query_names = metadata['query_types']
    results = list(result_dict['results'].values())
    if len(top_values) != len(query_names):
      top_values = [[] for _ in range(len(query_names))]

    for i in range(len(query_names)):
      top_values[i].extend([max([values['test_acc'] for values in query_iter]) for query_iter in results[i]])
  
  # Calculate metrics
  top_results = np.max(top_values, axis=1)
  means = np.mean(top_values, axis=1)
  variances = np.var(top_values, axis=1)
  if verbose:
    print('Top Values')
    for i in range(len(query_names)):
      print(f'{query_names[i]}: avg {means[i]}, var {variances[i]}, top {top_results[i]}')
  return top_results, means, variances, query_names


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('-r', '--run', help='run_id of the folder which contains the validation results')
  parser.add_argument('-f', '--file_name', help='filename of the the validation results json', default='val-results')
  args = parser.parse_args()

  get_average_values(args.run, [f'/{args.file_name}.json'], verbose=True)