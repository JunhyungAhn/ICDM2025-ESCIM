import torch
import torch.nn as nn

from typing import List
from common.layers import MultiHeadSelfAttention, CrossNetwork

class MultiLayerPerceptron(nn.Module):
  def __init__(self, input_dim: int, hidden_dims: List[int]):
    super().__init__()

    # Input layer
    layers = [nn.Linear(input_dim, hidden_dims[0]), nn.LeakyReLU()]

    # Hidden layers
    for i in range(len(hidden_dims) - 1):
      layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.LeakyReLU()]

    # Output layer
    layers.append(nn.Linear(hidden_dims[-1], 1))

    # Combine all layers
    self.mlp = nn.Sequential(*layers)
  
  def forward(self, inputs):
    return self.mlp(inputs)
  
class DeepFM(nn.Module):
  def __init__(self, input_dim, mlp_dims, dropout):
    super().__init__()
    self.input_dim = input_dim
    self.mlp_dims = mlp_dims
    self.dropout = dropout
    
    # Linear Part
    self.linear = nn.Linear(input_dim, 1)
    self.bias = nn.Parameter(torch.zeros(1,))
    
    # Deep Part
    self.deep_layers = nn.ModuleList()
    for mlp_dim in mlp_dims:
      self.deep_layers.append(nn.Linear(input_dim, mlp_dim))
      self.deep_layers.append(nn.BatchNorm1d(mlp_dim))
      self.deep_layers.append(nn.ReLU())
      self.deep_layers.append(nn.Dropout(dropout))
      input_dim = mlp_dim
    self.deep_layers.append(nn.Linear(input_dim, 1))
  
  def forward(self, inputs):
    # Linear Part
    linear_part = self.linear(inputs).sum(dim=1) + self.bias
    linear_part = linear_part.view(-1, 1)
    
    # FM Part: xy + yz + zx = 0.5 * (x+y+z)^2 - (x^2+y^2+z^2)
    fm_x = inputs
    square_of_sum = torch.sum(fm_x, dim=1) ** 2
    sum_of_square = torch.sum(fm_x ** 2, dim=1)
    fm_part = 0.5 * (square_of_sum.view(-1, 1) - sum_of_square.view(-1, 1)).sum(1, keepdim=True)
    
    # Deep Part
    deep_x = fm_x.view(-1, self.input_dim)
    for layer in self.deep_layers:
      deep_x = layer(deep_x)
  
    # Combine Parts
    result = linear_part + fm_part + deep_x
    
    return result
  
class AutoInt(nn.Module):
  def __init__(self, input_dim, num_heads, num_layers, dropout_rate):
    super().__init__()
    self.attention_layers = nn.ModuleList([
      MultiHeadSelfAttention(
        input_dim=input_dim,
        attention_dim=None,
        num_heads=num_heads,
        dropout_rate=dropout_rate
      ) for _ in range(num_layers)
    ])
    self.fc = nn.Linear(input_dim, 1)

  def forward(self, x):
    for attention_layer in self.attention_layers:
      x = attention_layer(x)
    x = x.mean(dim=1) # Pooling
    
    out = self.fc(x)
    
    return out
  
class DCNv2(nn.Module):
  def __init__(
    self, 
    input_dim: int,
    mlp_dims: List[int],
    cross_layer_num: int,
    dropout_rate: float
  ):
  
    super().__init__()
    self.cross_network = CrossNetwork(input_dim, cross_layer_num)
    self.deep_network = nn.Sequential()
    
    for i, mlp_dim in enumerate(mlp_dims):
      if i==0:
        self.deep_network.add_module(f'linear_{i}', nn.Linear(input_dim, mlp_dim))
      else:
        self.deep_network.add_module(f'linear_{i}', nn.Linear(mlp_dims[i-1], mlp_dim))
      self.deep_network.add_module('batchnorm_%d' % i, nn.BatchNorm1d(mlp_dim))
      self.deep_network.add_module('relu_%d' % i, nn.ReLU())
      self.deep_network.add_module('dropout_%d' % i, nn.Dropout(dropout_rate))
    self.deep_network.add_module('final_linear', nn.Linear(mlp_dims[-1], 1))
    
    self.final_linear = nn.Linear(input_dim+1, 1)

  def forward(self, x):
    cross_out = self.cross_network(x)
    deep_out = self.deep_network(x)
    total_out = torch.cat([cross_out, deep_out], dim=1)
    out = self.final_linear(total_out) 
    
    return out
  