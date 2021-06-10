from training.utils import check_accuracy, get_lr, add_pr_curves, save_model
from constants import ConfigManager

import torch
import torch.nn.functional as F

device = ConfigManager.device
USE_TB = ConfigManager.USE_TB

def train_part(model, optimizer, train_data, test_data, epochs=1, scheduler=None, scheduler_type='epoch', 
               warmup=0, writer=None, log_freq=1, log_level=2, loss_f=F.cross_entropy, log_start=0, 
               save_best_model=False, save_path=None):
  """    
  Inputs:
  - model: A PyTorch Module giving the model to train.
  - optimizer: An Optimizer object we will use to train the model
  - epochs: (Optional) A Python integer giving the number of epochs to train for
  - scheduler: (Optional) An lr_scheduler object which decreases the learning rate wrt the schedule
  - warmup: (Optional) A warmup period whereby the scheduler will not decrease the learning rate
  - writer: (Optional) A TensorBoard SummaryWriter which logs metrics based on log frequency
  - log_freq: The frequency to log training, testing accuracy and loss to both tensorboard and printed
  - log_level: 0 for Loss only, 1 for Loss and Train Accuracy, 2 for Loss, Train and Validation Accuracy
  - loss: Loss function, default is Cross Entropy (for logits), use NNL for log probability outputs

  Returns: Model accuracies during training.
  """
  if scheduler is not None and scheduler_type not in {'epoch', 'acc', 'train_acc', 'loss'}:
    raise ValueError('Scheduler update type not recognised. Current types are: [epoch, acc, train_acc, loss]')

  save_models = save_best_model and save_path is not None
  best_score = 0.0
  
  print("Number of train datapoints: %d" % (len(train_data)))
  model = model.to(device=device)  # move the model parameters to CPU/GPU
  hist = []
  for e in range(epochs):
    num_correct = 0
    num_samples = 0

    for t, (x, y) in enumerate(train_data):
      # Zero out all of the gradients for the variables which the optimizer will update.
      optimizer.zero_grad()
      
      model.train()  # put model to training mode
      x = x.to(device=device)  # move to device, e.g. GPU
      y = y.to(device=device, dtype=torch.long)

      scores = model(x)
      loss = F.cross_entropy(scores, y)

      _, preds = scores.max(1)
      num_correct += (preds == y).sum()
      num_samples += preds.size(0)
      loss.backward()

      # Update the parameters of the model using the gradients
      optimizer.step()

    train_acc = 100 * float(num_correct) / num_samples

    # Calculate metrics based on the log frequency
    if (log_freq > -1 and (e + 1) % log_freq == 0 and (e >= log_start)) or ((e + 1) == epochs):
      hist.append({'epoch': e, 'lr': get_lr(optimizer), 'loss': loss.item()})
      # Checks the loss, test accuracy and train accuracy based on log level
      if log_level > 0:
        hist[-1]['train_acc'] = train_acc
        
      if log_level == 2:
        acc = check_accuracy(test_data, model, verbose=False)
        hist[-1]['test_acc'] = acc

      # Print to stdout
      if log_level == 0:
        print('[%.2d/%.2d] lr %.5f | loss = %.4f' % (e + 1, epochs, get_lr(optimizer), loss.item()))
      elif log_level == 1:
        print('[%.2d/%.2d] lr %.5f | loss = %.4f | train acc = %.2f' \
              % (e + 1, epochs, get_lr(optimizer), loss.item(), train_acc))
      else:
        print('[%.2d/%.2d] lr %.5f | loss = %.4f | train acc = %.2f | test acc = %.2f' \
              % (e + 1, epochs, get_lr(optimizer), loss.item(), train_acc, acc))

      # Save models enabled
      if save_models and log_level == 2:
        if acc >= best_score:
          best_score = acc
          save_model(model, save_path, verbose=True)

      # Write to tensorboard if available
      if USE_TB and not writer is None:
        if type(writer) is tuple:
          loss_w, train_w, test_w = writer
          loss_w.add_scalar("Curves/loss", loss.item(), e)

          if log_level > 0:
            train_w.add_scalar("Curves/accuracy", train_acc, e)

          if log_level > 1:
            test_w.add_scalar("Curves/accuracy", acc, e)

            if (e + 1) == epochs:
              add_pr_curves(test_data, model, test_w)

        else:
          writer.add_scalar("Curves/loss", loss.item(), e)

          if log_level > 0:
            writer.add_scalar("Curves/accuracy", train_acc, e)

          if log_level > 1:
            writer.add_scalar("Curves/accuracy", acc, e)
    
    # Step through the scheduler if after warmup period
    if not scheduler == None and e >= warmup:
      if scheduler_type == 'epoch':
        scheduler.step(e)
      elif scheduler_type == 'train_acc':
        scheduler.step(train_acc)
      elif scheduler_type == 'loss':
        scheduler.step(loss.item())
      else:
        raise ValueError('Scheduler update type not recognised. Current types are: [epoch, acc, train_acc, loss]')

  return hist