import os
import fire
import json
import torch

from Ali_Express.train import train
from Ali_Express.dataset import AliExpressDataset, get_dataloader

def run(config_file="Ali_Express/config.json", country='ES', **kwargs):
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
  
  train_dataset = AliExpressDataset(os.path.join('./Ali_Express/data', country) + '/train.csv')
  
  # Generate dataloaders
  train_loader = get_dataloader(
    os.path.join('./Ali_Express/data/', country) + '/train.csv',
    batch_size, 
    is_train=True
  )

  train_clicked_loader = get_dataloader(
    os.path.join('./Ali_Express/data/', country) + '/train_clicked.csv',
    batch_size, 
    is_train=True
  )

  val_loader = get_dataloader(
    os.path.join('./Ali_Express/data/', country) + '/val.csv',
    batch_size, 
    is_train=False
  )

  test_loader = get_dataloader(
    os.path.join('./Ali_Express/data/', country) + '/test.csv',
    batch_size, 
    is_train=False
  )

  categorical_field_dims = train_dataset.field_dims
  numerical_num = train_dataset.numerical_num
  
  device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
  
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
  
  train(config, train_loader, train_clicked_loader, val_loader, test_loader, categorical_field_dims, numerical_num, device)
  
if __name__ == '__main__':
  fire.Fire(run)