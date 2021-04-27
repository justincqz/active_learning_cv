from torch import float32, device, cuda
import os
import socket
import atexit
import subprocess

def port_check(HOST, PORT):
   s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   s.settimeout(2) # Timeout in case of port not open
   try:
      s.connect((HOST, PORT))
      return True
   except:
      return False

class ConfigManager():
  # Flags
  USE_RAPIDS = False
  USE_TB = True
  USE_GPU = False
  USE_MDE = False
  USE_VALIDATION = False
  SAVE_LOC = './results'
  LOG_DIR  = f'{SAVE_LOC}/tensorboard'
  
  # Constants
  dtype = float32
  device = device('cuda') if cuda.is_available() else device('cpu')
  
  # Attempt to enable the tensorboard
  @classmethod
  def load_tensorboard(self):
    if port_check("127.0.1.1", 6006):
      return
    
    # Ensure that the tensorboard directory exists
    os.makedirs(self.LOG_DIR, exist_ok=True)
    
    # Start the tensorboard process
    p = subprocess.Popen(f'tensorboard --logdir {self.LOG_DIR} --host 127.0.0.1 --port 6006', shell=False)
    
    # Handle cleanup during script exit
    def cleanup_tensorboard():
      p.terminate()
    atexit.register(cleanup_tensorboard)
    print('Access tensorboard at localhost:6006')
