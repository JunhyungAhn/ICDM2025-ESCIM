import torch
import torch.nn as nn

from common.backbone_models import MultiLayerPerceptron, DeepFM, AutoInt, DCNv2
from common.layers import EmbeddingLayer
      
class CtrTower(nn.Module):
  """NN for CTR prediction"""

  def __init__(self, input_dim, backbone_model, model_config):
    super().__init__()
    
    if backbone_model=='mlp':
      mlp_dims = model_config['mlp_dims']
      self.model = MultiLayerPerceptron(input_dim, mlp_dims)
    
    elif backbone_model=='deepfm':
      mlp_dims = model_config['mlp_dims']
      dropout = model_config['dropout']
      self.model = DeepFM(input_dim, mlp_dims, dropout)
    
    elif backbone_model=='autoint':
      num_heads = model_config['num_heads']
      num_layers = model_config['num_layers']
      dropout = model_config['dropout']
      self.model = AutoInt(input_dim, num_heads, num_layers, dropout)
    
    elif backbone_model=='dcnv2':
      mlp_dims = model_config['mlp_dims']
      cross_layer_num = model_config['cross_layer_num']
      dropout = model_config['dropout']
      self.model = DCNv2(input_dim, mlp_dims, cross_layer_num, dropout)
    
    else:
      raise Exception('Invalid Backbone Model')
    
    self.sigmoid = nn.Sigmoid()

  def forward(self, inputs):
    p = self.model(inputs)
    
    return self.sigmoid(p)

class CvrTower(nn.Module):
  """NN for CVR prediction"""

  def __init__(self, input_dim, backbone_model, model_config):
    super().__init__()
    
    if backbone_model=='mlp':
      mlp_dims = model_config['mlp_dims']
      self.model = MultiLayerPerceptron(input_dim, mlp_dims)
    
    elif backbone_model=='deepfm':
      mlp_dims = model_config['mlp_dims']
      dropout = model_config['dropout']
      self.model = DeepFM(input_dim, mlp_dims, dropout)
    
    elif backbone_model=='autoint':
      num_heads = model_config['num_heads']
      num_layers = model_config['num_layers']
      dropout = model_config['dropout']
      self.model = AutoInt(input_dim, num_heads, num_layers, dropout)
    
    elif backbone_model=='dcnv2':
      mlp_dims = model_config['mlp_dims']
      cross_layer_num = model_config['cross_layer_num']
      dropout = model_config['dropout']
      self.model = DCNv2(input_dim, mlp_dims, cross_layer_num, dropout)
    
    else:
      raise Exception('Invalid Backbone Model')
    
    self.sigmoid = nn.Sigmoid()

  def forward(self, inputs):            
    p = self.model(inputs)
    
    return self.sigmoid(p)


class ImpTower(nn.Module):
    """NN for Imputation"""

    def __init__(self, input_dim, backbone_model, model_config):
      super().__init__()
      if backbone_model=='mlp':
        mlp_dims = model_config['mlp_dims']
        self.model = MultiLayerPerceptron(input_dim, mlp_dims)
      
      elif backbone_model=='deepfm':
        mlp_dims = model_config['mlp_dims']
        dropout = model_config['dropout']
        self.model = DeepFM(input_dim, mlp_dims, dropout)
      
      elif backbone_model=='autoint':
        num_heads = model_config['num_heads']
        num_layers = model_config['num_layers']
        dropout = model_config['dropout']
        self.model = AutoInt(input_dim, num_heads, num_layers, dropout)
      
      elif backbone_model=='dcnv2':
        mlp_dims = model_config['mlp_dims']
        cross_layer_num = model_config['cross_layer_num']
        dropout = model_config['dropout']
        self.model = DCNv2(input_dim, mlp_dims, cross_layer_num, dropout)
      
      else:
        raise Exception('Invalid Backbone Model')
      
      self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):            
      p = self.model(inputs)
      
      return self.sigmoid(p) 
    
class Escm2(nn.Module):
    def __init__(
      self,
      categorical_field_dims, 
      numerical_num,
      embedding_size: int = 5,
      backbone_model: str = 'mlp',
      model_config: dict = {},
      regularizer: str = 'dr'
    ):
    
      super().__init__()
      
      self.embedding_size = embedding_size
      
      self.embedding = EmbeddingLayer(categorical_field_dims, embedding_size)
      self.numerical_layer = torch.nn.Linear(numerical_num, embedding_size)
      self.embed_output_dim = (len(categorical_field_dims) + 1) * embedding_size
      
      self.regularizer = regularizer
      
      self.input_dim = (len(categorical_field_dims) + 1) * embedding_size
      
      self.ctr_model = CtrTower(self.input_dim, backbone_model, model_config)
      self.cvr_model = CvrTower(self.input_dim, backbone_model, model_config)
      self.imp_model = ImpTower(self.input_dim, backbone_model, model_config)

    def forward(self, categorical_x, numerical_x):
      # Concat outputs of embedding layer
      categorical_emb = self.embedding(categorical_x)
      numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
      emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim) 
          
      # Predict pCTR, pCVR
      pctr = self.ctr_model(emb)
      pcvr = self.cvr_model(emb)
      pctcvr = torch.mul(pctr, pcvr)
      
      if self.regularizer=='dr':
        pimp = self.imp_model(emb)
      else:
        pimp = None
      
      output = {'pctr': pctr, 'pcvr': pcvr, 'pctcvr': pctcvr, 'pimp': pimp}
      
      return output
