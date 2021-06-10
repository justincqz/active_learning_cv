from avg_validation_stats import get_average_values
from ast import literal_eval
import matplotlib.pyplot as plt

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('-r', '--run', help='run_id of the folder which contains the validation results')
  args = parser.parse_args()

  target_folder = './results/runs/500002' if args.run is None else f'./results/runs/{args.run}'
  # target_folder = './results/runs/500002'
  target_file = '/results.json'
  metadata_file = '/run_details.json'

  with open(target_folder + metadata_file, 'r') as f:
    metadata = ''.join(f.readlines())
  metadata = literal_eval(metadata)

  with open(target_folder + target_file, 'r') as f:
    raw = ''.join(f.readlines())
  result_dict = literal_eval(raw)
  iters = result_dict['iter']
  query_names = metadata['query_types']
  results = list(result_dict['results'].values())

  # Plot the final test accuracies
  final_test_accs = []
  top_values = []
  for i in range(len(query_names)):
    top_values.append([max([values['test_acc'] for values in query_iter]) for query_iter in results[i]])

  print('Top Values')
  for i in range(len(query_names)):
    print(f'{query_names[i]}: {top_values[i]}')
  
  _, mean_vals, _, _ = get_average_values(args.run)
  for i in range(len(top_values)):
    top_values[i][-1] = mean_vals[i]

  plt.figure()
  plt.title('Cifar10 with ResNet20')
  sampled_amounts = [(i + 1) * 0.1 for i in range(len(top_values[0]))]
  for i in range(len(query_names)):
    q_name = query_names[i]
    if len(sampled_amounts) != len(top_values[i]):
      print(f'{q_name}: Failed to get iteration score')
      continue
    plt.plot(sampled_amounts, top_values[i], label=q_name)
    print(f'{q_name}: {top_values[i][-1]}')
  plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
  plt.legend()
  plt.show()