import os
import json
import fire
import torch

from Ali_CCP.train import train
from Ali_CCP.constants import TRAIN_PATH, TRAIN_CLICKED_PATH, VAL_PATH, TEST_PATH
from Ali_CCP.dataset import get_dataloader

def run(config_file="Ali_CCP/config.json", **kwargs):
  os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
  
  # Read configuration file
  config = {}
  if config_file:
    with open(config_file, 'r') as f:
      config.update(json.load(f))
  config.update(kwargs)
  
  # Define parameters
  batch_size = config['batch_size']
  gpu_num = config['gpu_num']
  device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'

  train_loader = get_dataloader(TRAIN_PATH, batch_size, is_train=True)
  train_clicked_loader = get_dataloader(TRAIN_CLICKED_PATH, batch_size, is_train=True)
  val_loader = get_dataloader(VAL_PATH, batch_size, is_train=False)
  test_loader = get_dataloader(TEST_PATH, batch_size, is_train=False)
  
  # Define core model config
  backbone_model = config["backbone_model"]
  mlp_dims = config["mlp_dims"]
  dropout = config["dropout"]

  if backbone_model=='mlp':
    model_config = {
      'mlp_dims': mlp_dims,
      'dropout': dropout
    }
    
  elif backbone_model=='deepfm':
    model_config = {
      'mlp_dims': mlp_dims,
      'dropout': dropout,
    }
    
  elif backbone_model=='autoint':
    model_config = {
      'num_heads': 2,
      'num_layers': 2,
      'dropout': dropout,
    }
    
  elif backbone_model=='dcnv2':
    model_config = {
      'mlp_dims': mlp_dims,
      'dropout': dropout,
      'cross_layer_num': 2
    }
    
  else:
    raise Exception("Invalid Backbone Model")
  
  config['model_config'] = model_config
  
  train(config, train_loader, train_clicked_loader, val_loader, test_loader, device)
  
if __name__ == '__main__':
  fire.Fire(run)