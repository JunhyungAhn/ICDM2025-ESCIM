import torch

import numpy as np
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
  def __init__(self, dropout_rate: float = 0.0):
    super().__init__()
    self.dropout = None
    if dropout_rate > 0:
      self.dropout = nn.Dropout(dropout_rate)
  
  def forward(self, Q, K, V, scale=None, mask=None):
    scores = torch.matmul(Q, K.transpose(-1, -2))
    
    if scale:
      scores = scores / scale
    if mask:
      scores = scores.masked_fill_(mask, -1e-10)
    
    attention = scores.softmax(dim=-1)
    if self.dropout is not None:
      attention = self.dropout(attention)
      
    output = torch.matmul(attention, V)
    
    return output, attention
  
class MultiHeadSelfAttention(nn.Module):
  def __init__(
    self, 
    input_dim: int,
    attention_dim: int = None,
    num_heads: int = 1,
    dropout_rate: float = 0.0,
    use_res: bool = True,
    use_scale: bool = False,
    layer_norm: bool = False
  ):

    super().__init__()
    
    if attention_dim is None:
      attention_dim = input_dim
    if num_heads <= 0:
      raise ValueError('num_head must be a int > 0')
    assert attention_dim % num_heads == 0, "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
    
    self.input_dim = input_dim
    self.head_dim = attention_dim // num_heads
    self.num_heads = num_heads
    self.use_res = use_res
    self.scale = self.head_dim ** 0.5 if use_scale else None

    self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
    self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
    self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
    
    if self.use_res and input_dim != attention_dim:
      self.W_res = nn.Linear(input_dim, attention_dim, bias=False)
    else:
      self.W_res = None
      
    self.dot_attention = ScaledDotProductAttention(dropout_rate)
    
    self.layer_norm = nn.LayerNorm(attention_dim) if layer_norm else None
    
    self.fc_out = nn.Linear(input_dim, input_dim)
            
  def forward(self, x):
    residual = x
    
    query, key, value = self.W_q(x), self.W_k(x), self.W_v(x)
    
    # split by heads
    batch_size = query.size(0)
    query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    
    output, _ = self.dot_attention(query, key, value, scale=self.scale)
    
    output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads*self.head_dim)
    
    if self.W_res:
      residual = self.W_res(residual)
        
    if self.use_res:
      residual = residual.view(batch_size, -1, self.num_heads*self.head_dim)
      output += residual
    
    if self.layer_norm:
      output = self.layer_norm(output)
    
    return output
  
class CrossNetwork(nn.Module):
  def __init__(self, input_dim: int, layer_num: int):
    super().__init__()
    
    self.layer_num = layer_num
    self.cross_weights = nn.ParameterList([nn.Parameter(torch.randn(input_dim, 1)) for _ in range(layer_num)])    
    self.cross_biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(layer_num)])    
  
  def forward(self, x):
    x0 = x.clone()
    for i in range(self.layer_num):
      # Compute cross product
      xw = torch.matmul(x, self.cross_weights[i]) # batch_size x 1
      cross = x0 * xw # (batch_size, input_dim)
      # Add bias and input
      x = cross + self.cross_biases[i] + x # In-place addition for memory efficiency
      
    return x
  
class EmbeddingLayer(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)
      